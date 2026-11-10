"""Static native MLX contracts; runtime imports remain lazy."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from typing import Protocol

    import mlx.core as mx
    from medmlx_core.typing import MlxRuntime
    from medmlx_core.volume_workspace import VolumeWorkspace

    from swin_unetr_mlx._arrays import Float32Array

    type Array = mx.array

    type ArrayTree = Array | Sequence[ArrayTree] | Mapping[str, ArrayTree]

    class MLX(MlxRuntime, Protocol):
        def load(self, file: str, /) -> Array | dict[str, Array]:
            """Load a native array or named tensor mapping from a file."""
            ...

        def save_safetensors(self, file: str, arrays: dict[str, Array]) -> None:
            """Write named MLX arrays to a safetensors file."""
            ...

        def eval(self, *values: ArrayTree) -> None:
            """Evaluate arrays contained in the supplied native trees."""
            ...

        def full(
            self, shape: Sequence[int], vals: int | float, dtype: mx.Dtype | None = None
        ) -> Array:
            """Create an array filled with a scalar value."""
            ...

        def softmax(self, x: Array, /, axis: int = -1) -> Array:
            """Normalize logits along the selected axis."""
            ...

        def roll(self, x: Array, shift: Sequence[int], axis: Sequence[int]) -> Array:
            """Roll an array along the selected axes."""
            ...

        def erf(self, x: Array, /) -> Array:
            """Apply the error function elementwise."""
            ...

        def synchronize(self) -> None:
            """Wait for outstanding work on the default MLX stream."""
            ...

    class SlidingWindowInference(Protocol):
        def __call__(
            self,
            inputs: Float32Array,
            roi_size: Sequence[int],
            sw_batch_size: int,
            predictor: Callable[[Array], Array],
            *,
            overlap: float,
            with_coord: bool = False,
            workspace: VolumeWorkspace | None = None,
        ) -> Float32Array:
            """Run overlapping window inference and return combined float32 outputs."""
            ...

    class CoreSlidingWindow(Protocol):
        sliding_window_inference: SlidingWindowInference

