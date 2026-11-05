"""Synchronized FP32 network-window inference."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from swin_unetr_mlx._arrays import Float32Array, HostArray

if TYPE_CHECKING:
    from swin_unetr_mlx.network import SwinUNETR
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory


def forward_patch(network: SwinUNETR, values: HostArray) -> Float32Array:
    """Run one prepared float32 window and return raw logits.

    The input is NCDHW ``(B, 1, 96, 96, 96)`` and the result float32
    ``(B, 14, 96, 96, 96)``; no intensity scaling, softmax or spatial transform is
    applied.

    Raises:
        InvalidInputError: If ``values`` is not finite float32.
        RuntimeError: If the output is not finite float32.
    """
    values = np.asarray(values)
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise InvalidInputError("Network input must be finite float32")
    mx = network.mx
    result = network(cast(ArrayFactory, mx.array)(values))
    mx.eval(result)
    mx.synchronize()
    host = cast(Float32Array, np.asarray(result))
    if host.dtype != np.float32 or not np.isfinite(host).all():
        raise RuntimeError("Network produced invalid float32 outputs")
    return host
