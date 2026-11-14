"""Abdominal CT organ segmentation with SwinUNETR in native MLX.

``segment`` runs one CT end to end. Names load on first access, so importing the
package does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swin_unetr_mlx.inference import forward_patch as forward_patch
    from swin_unetr_mlx.medmlx import segment as segment
    from swin_unetr_mlx.network import SwinUNETR as SwinUNETR
    from swin_unetr_mlx.weights import load_network as load_network

from swin_unetr_mlx.specification import specification

_EXPORTS = {
    "SwinUNETR": "swin_unetr_mlx.network",
    "forward_patch": "swin_unetr_mlx.inference",
    "load_network": "swin_unetr_mlx.weights",
    "segment": "swin_unetr_mlx.medmlx",
}

__version__ = "0.2.7"
__all__ = [
    "SwinUNETR",
    "forward_patch",
    "load_network",
    "segment",
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
