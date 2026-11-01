"""Native MLX SwinUNETR; network names load on first access."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swin_unetr_mlx.inference import forward_patch as forward_patch
    from swin_unetr_mlx.network import SwinUNETR as SwinUNETR
    from swin_unetr_mlx.weights import load_model as load_model

from swin_unetr_mlx.specification import specification

_EXPORTS = {
    "SwinUNETR": "swin_unetr_mlx.network",
    "forward_patch": "swin_unetr_mlx.inference",
    "load_model": "swin_unetr_mlx.weights",
}

__version__ = "0.2.6"
__all__ = [
    "SwinUNETR",
    "forward_patch",
    "load_model",
    "specification",
]


def __getattr__(name: str) -> object:
    """Load a public export from its module on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'swin_unetr_mlx' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and lazy package exports."""
    return sorted({*globals(), *_EXPORTS})
