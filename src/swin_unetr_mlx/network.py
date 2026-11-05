"""Native FP32 SwinUNETR with explicit MONAI patch-merging profiles."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from math import prod
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX, Array

import numpy as np
from medmlx_core.channel_last import ChannelLastGraph
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory

from swin_unetr_mlx.runtime import require_metal

type MergingProfile = Literal["monai-1.4", "monai-1.6"]

# Output depth planes per same-padded 3D convolution slab (see ``SwinUNETR.conv``).
_DEPTH_SLAB = 12


class SwinUNETR(ChannelLastGraph):
    """FP32 SwinUNETR forward pass for one 96^3 CT window.

    ``__call__`` takes a float32 ``(B, 1, 96, 96, 96)`` array (channels-first, CT already
    clipped to -175..250 HU and scaled to [0, 1]) and returns raw logits
    ``(B, 14, 96, 96, 96)`` without softmax. Internally tensors are channels-last
    ``(B, D, H, W, C)``, MLX's native convolution layout. ``weights`` maps converted
    checkpoint tensor names to MLX arrays. ``merging_profile`` selects the
    patch-merging phase order of the MONAI release being matched (see ``merge``).
    Raises ``MissingDependencyError`` if the MLX Metal GPU is unavailable and
    ``ValueError`` for a wrongly shaped or non-float32 input.
    """

    mx: MLX
    w: Mapping[str, Array]
    # These methods are inherited from medmlx-core's ChannelLastGraph.
    # Static declarations retain the original runtime implementations.
    if TYPE_CHECKING:

        def linear(self, x: Array, name: str) -> Array:
            """Apply the inherited linear layer using named weights."""
            ...
        def layer_norm(self, x: Array, name: str | None = None) -> Array:
            """Apply inherited layer normalization using optional named weights."""
            ...
        def residual3d(self, x: Array, name: str) -> Array:
            """Apply an inherited three-dimensional residual block."""
            ...
        def deconv3d(self, x: Array, name: str) -> Array:
            """Apply an inherited three-dimensional transposed convolution."""
            ...
        def require_input(self, x: Array, shape: tuple[int, ...]) -> None:
            """Validate native input dtype and expected spatial dimensions."""
            ...

    def __init__(
        self,
        weights: Mapping[str, Array],
        *,
        mx: MLX,
        merging_profile: MergingProfile = "monai-1.4",
    ) -> None:
        """Initialize the native network with weights and a supported merging profile."""
        if merging_profile not in {"monai-1.4", "monai-1.6"}:
            raise InvalidInputError(f"Unsupported Swin patch-merging profile: {merging_profile}")
        require_metal(mx)
        super().__init__(weights, mx=mx)
        self.merging_profile = merging_profile
        self._attention_masks: dict[
            tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]], Array
        ] = {}

    def conv(
        self, x: Array, name: str, *, padding: int = 0, stride: int = 1, groups: int = 1
    ) -> Array:
        """Apply a named convolution, splitting deep same-padded 3D convolutions into depth slabs.

        MLX 0.32 runs a 3x3x3 convolution with nonzero depth padding on a much slower Metal
        path than one with zero depth padding. Each slab of ``_DEPTH_SLAB`` output planes
        convolves its input planes plus a one-plane halo (zeros at the volume boundary) with
        depth padding 0, roughly doubling throughput for the 96^3 decoder blocks. The slab
        size keeps that path's scratch memory below the encoder's attention peak. Logits
        stay within the seeded PyTorch qualification bound but are not bitwise identical
        to the unsplit convolution, whose reduction order differs.
        """
        mx = self.mx
        weight = self.w[f"{name}.weight"]
        depth = x.shape[1]
        if weight.ndim != 5 or padding != 1 or stride != 1 or groups != 1 or depth <= _DEPTH_SLAB:
            return super().conv(x, name, padding=padding, stride=stride, groups=groups)
        weight = mx.transpose(weight, (0, 2, 3, 4, 1))
        y = mx.zeros((*x.shape[:4], weight.shape[0]), dtype=x.dtype)
        for start in range(0, depth, _DEPTH_SLAB):
            stop = min(start + _DEPTH_SLAB, depth)
            part = x[:, max(start - 1, 0) : min(stop + 1, depth)]
            halo = (int(start == 0), int(stop == depth))
            if any(halo):
                part = mx.pad(part, [(0, 0), halo, (0, 0), (0, 0), (0, 0)])
            y[:, start:stop] = mx.conv3d(part, weight, padding=(0, 1, 1))
        bias = self.w.get(f"{name}.bias")
        return y if bias is None else y + bias

    @staticmethod
    def partition(x: Array, window: tuple[int, ...]) -> Array:
        """Split ``(B, D, H, W, C)`` into ``(B * n_windows, prod(window), C)``.

        Each spatial extent must be divisible by its corresponding window size;
        the caller pads beforehand when necessary.
        """
        import mlx.core as mx

        b, d, h, w, c = x.shape
        wd, wh, ww = window
        return mx.reshape(
            mx.transpose(
                mx.reshape(x, (b, d // wd, wd, h // wh, wh, w // ww, ww, c)),
                (0, 1, 3, 5, 2, 4, 6, 7),
            ),
            (-1, prod(window), c),
        )

    def reverse(self, x: Array, window: tuple[int, ...], shape: tuple[int, ...]) -> Array:
        """Inverse of ``partition`` back to ``shape`` = ``(B, D, H, W, C)``."""
        b, d, h, w, c = shape
        wd, wh, ww = window
        mx = self.mx
        return mx.reshape(
            mx.transpose(
                mx.reshape(x, (b, d // wd, h // wh, w // ww, wd, wh, ww, c)),
                (0, 1, 4, 2, 5, 3, 6, 7),
            ),
            shape,
        )

    def mask(
        self, shape: tuple[int, ...], window: tuple[int, ...], shift: tuple[int, ...]
    ) -> Array:
        """Additive attention mask ``(n_windows, N, N)`` for shifted windows.

        Token pairs from different shift regions get -100 (as in MONAI), others 0.
        Masks depend only on window geometry and are retained for this network's
        lifetime; the fixed 96^3 graph uses three geometries across all batches.
        Callers must not modify the shared result.
        """
        key = (shape, window, shift)
        cached = self._attention_masks.get(key)
        if cached is not None:
            return cached
        values = np.zeros((1, *shape, 1), dtype=np.float32)
        slices = [
            (slice(-w), slice(-w, -s), slice(-s, None)) for w, s in zip(window, shift, strict=True)
        ]
        for i, region in enumerate(product(*slices)):
            values[(slice(None), *region, slice(None))] = i
        values = self.partition(cast(ArrayFactory, self.mx.array)(values), window)[..., 0]
        mask = self.mx.where(values[:, None, :] == values[:, :, None], 0.0, -100.0)
        self._attention_masks[key] = mask
        return mask

    def attention(self, x: Array, name: str, heads: int, mask: Array | None) -> Array:
        """Apply window self-attention with relative-position bias and an optional shift mask.

        ``x`` is ``(window_batches, tokens, channels)``; ``heads`` must divide
        channels. The result retains that shape. A mask has one ``(tokens,
        tokens)`` matrix per spatial window and broadcasts across image batches.
        """
        mx = self.mx
        b, n, c = x.shape
        qkv = mx.reshape(self.linear(x, f"{name}.qkv"), (b, n, 3, heads, c // heads))
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * (c // heads) ** -0.5, qkv[1], qkv[2]
        score = q @ mx.swapaxes(k, -2, -1)
        index = mx.reshape(self.w[f"{name}.relative_position_index"][:n, :n], (-1,))
        bias = self.w[f"{name}.relative_position_bias_table"][index]
        score = score + mx.transpose(mx.reshape(bias, (n, n, heads)), (2, 0, 1))[None]
        if mask is not None:
            nw = mask.shape[0]
            score = mx.reshape(score, (b // nw, nw, heads, n, n)) + mask[None, :, None]
            score = mx.reshape(score, (b, heads, n, n))
        y = mx.reshape(mx.transpose(mx.softmax(score, axis=-1) @ v, (0, 2, 1, 3)), (b, n, c))
        return self.linear(y, f"{name}.proj")

    def block(self, x: Array, name: str, heads: int, shifted: bool) -> Array:
        """Apply one window-attention and MLP residual block to an NDHWC feature map.

        Each spatial window is at most seven voxels wide; shifted blocks roll by
        three where that axis exceeds seven. Padding is removed before the first
        residual addition, so the result has the input shape.
        """
        mx = self.mx
        spatial = tuple(x.shape[1:4])
        # Window size 7 and shift 3; a dimension no larger than the window gets a window
        # of its own size and no shift (MONAI's get_window_size).
        window = tuple(min(7, v) for v in spatial)
        shift = tuple(3 if shifted and v > 7 else 0 for v in spatial)
        padding = tuple((w - v % w) % w for v, w in zip(spatial, window, strict=True))
        y = mx.pad(
            self.layer_norm(x, f"{name}.norm1"), [(0, 0), *((0, p) for p in padding), (0, 0)]
        )
        shape = tuple(y.shape)
        mask = None
        if any(shift):
            y = mx.roll(y, tuple(-s for s in shift), axis=(1, 2, 3))
            mask = self.mask(shape[1:4], window, shift)
        y = self.attention(self.partition(y, window), f"{name}.attn", heads, mask)
        y = self.reverse(y, window, shape)
        if any(shift):
            y = mx.roll(y, shift, axis=(1, 2, 3))
        x = x + y[:, : spatial[0], : spatial[1], : spatial[2], :]
        y = self.linear(self.layer_norm(x, f"{name}.norm2"), f"{name}.mlp.linear1")
        y = 0.5 * y * (1 + mx.erf(y * (2**-0.5)))  # exact (erf) GELU, as torch.nn.GELU
        return x + self.linear(y, f"{name}.mlp.linear2")

    def merge(self, x: Array, name: str) -> Array:
        """Concatenate eight sampled 2x2x2 phases, then normalize and reduce.

        The legacy 1.4 profile repeats some phases; 1.6 uses all eight distinct
        phases. Odd spatial dimensions are padded on their high sides first.
        """
        x = self.mx.pad(x, [(0, 0), *((0, v % 2) for v in x.shape[1:4]), (0, 0)])
        # MONAI 1.6 corrected the legacy 'merging' phases. The released 1.4
        # bundle retains its repeated phases unless the caller selects 1.6.
        phases = (
            (0, 0, 0),
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
        )
        phases += (
            ((1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
            if self.merging_profile == "monai-1.6"
            else ((1, 0, 1), (0, 1, 0), (0, 0, 1), (1, 1, 1))
        )
        x = self.mx.concatenate([x[:, i::2, j::2, k::2, :] for i, j, k in phases], axis=-1)
        return self.linear(self.layer_norm(x, f"{name}.norm"), f"{name}.reduction")

    def __call__(self, x: Array) -> Array:
        """Run the network on an admitted float32 window using Metal."""
        require_metal(self.mx)
        self.require_input(x, (1, 96, 96, 96))
        # MLX convolutions are channels-last: NCDHW in, NDHWC inside, NCDHW out.
        x = self.mx.transpose(x, (0, 2, 3, 4, 1))
        hidden = self.conv(x, "swinViT.patch_embed.proj", stride=2)
        states = [self.layer_norm(hidden)]
        for level in range(1, 5):
            name = f"swinViT.layers{level}.0"
            for block in range(2):
                hidden = self.block(
                    hidden, f"{name}.blocks.{block}", 3 * 2 ** (level - 1), bool(block)
                )
            hidden = self.merge(hidden, f"{name}.downsample")
            states.append(self.layer_norm(hidden))
        skips = [self.residual3d(x, "encoder1.layer")]
        skips.extend(self.residual3d(states[i], f"encoder{i + 2}.layer") for i in range(3))
        skips.append(states[3])
        hidden = self.residual3d(states[4], "encoder10.layer")
        for level, skip in zip(range(5, 0, -1), reversed(skips), strict=True):
            hidden = self.deconv3d(hidden, f"decoder{level}.transp_conv.conv")
            hidden = self.residual3d(
                self.mx.concatenate([hidden, skip], axis=-1), f"decoder{level}.conv_block"
            )
        return self.mx.transpose(self.conv(hidden, "out.conv.conv"), (0, 4, 1, 2, 3))
