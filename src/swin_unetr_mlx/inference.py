"""Synchronized FP32 network-window inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, overload

import numpy as np

from swin_unetr_mlx._arrays import Float32Array, HostArray

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX, Array
    from swin_unetr_mlx.network import SwinUNETR
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory


class WindowNetwork(Protocol):
    """Define the native network contract evaluated by ``forward_patch``."""

    mx: MLX

    def __call__(self, values: Array, /) -> Array | dict[str, Array]:
        """Run a native window network and return logits or named outputs."""
        ...


@overload
def forward_patch(network: SwinUNETR, values: HostArray) -> Float32Array:
    """Run one SwinUNETR window and return host logits."""
    ...
@overload
def forward_patch(
    network: WindowNetwork, values: HostArray
) -> Float32Array | dict[str, Float32Array]:
    """Run one window and return host logits or named outputs."""
    ...
def forward_patch(
    network: WindowNetwork, values: HostArray
) -> Float32Array | dict[str, Float32Array]:
    """Run one prepared float32 window and return raw logits.

    For ``SwinUNETR`` the input is NCDHW ``(B, 1, 96, 96, 96)`` and the result float32
    ``(B, 14, 96, 96, 96)``; no intensity scaling, softmax or spatial transform is
    applied. A network returning a dict of outputs yields a dict of float32 arrays.

    Raises:
        InvalidInputError: If ``values`` is not finite float32.
        RuntimeError: If an output is not finite float32.
    """
    values = np.asarray(values)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise InvalidInputError("Network input must be finite float32")
    mx = network.mx
    result = network(cast(ArrayFactory, mx.array)(values))
    mx.eval(result)
    mx.synchronize()
    host = (
        {name: cast(Float32Array, np.asarray(value)) for name, value in result.items()}
        if isinstance(result, dict)
        else cast(Float32Array, np.asarray(result))
    )
    outputs = host.values() if isinstance(host, dict) else (host,)
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in outputs):
        raise RuntimeError("Network produced invalid float32 outputs")
    return host
