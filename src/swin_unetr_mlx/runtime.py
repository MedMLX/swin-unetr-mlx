"""Apple Silicon admission before the shared MLX runtime is loaded."""

from __future__ import annotations

import platform
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX

from medmlx_core import runtime as core_runtime
from medmlx_core.errors import MissingDependencyError

_HINT = "Install swin-unetr-mlx on macOS Apple Silicon with an available Metal GPU."


def _require_apple_silicon() -> None:
    """Reject hosts other than macOS on Apple Silicon."""
    system, machine = platform.system(), platform.machine()
    if (system, machine) != ("Darwin", "arm64"):
        raise MissingDependencyError(
            f"SwinUNETR requires macOS arm64 with Metal; this host is {system} {machine}",
            hint=_HINT,
        )


def require_metal(mx: MLX) -> None:
    """Reject unavailable Metal or an execution device other than the Metal GPU."""
    _require_apple_silicon()
    if not mx.metal.is_available():
        raise MissingDependencyError("MLX Metal GPU is not available", hint=_HINT)
    if mx.default_device().type != mx.gpu:
        raise MissingDependencyError("SwinUNETR requires the MLX Metal GPU device", hint=_HINT)


def import_mlx() -> MLX:
    """Load MLX through medmlx-core after admitting only the supported host."""
    _require_apple_silicon()
    try:
        core_runtime.import_mlx()
        import mlx.core as native_mx

        mx = cast("MLX", native_mx)
    except MissingDependencyError as error:
        raise MissingDependencyError(str(error), hint=_HINT) from error
    require_metal(mx)
    return mx
