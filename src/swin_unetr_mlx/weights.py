"""Strict standalone safetensors loading."""

from __future__ import annotations

from pathlib import Path

from medmlx_core.errors import AssetNotReadyError
from medmlx_core.weights import load_tensors

from swin_unetr_mlx.network import MergingProfile, SwinUNETR
from swin_unetr_mlx.runtime import import_mlx
from swin_unetr_mlx.specification import specification

MODEL_ID = "swin-unetr"

# Indices address the (2 * 7 - 1) ** 3 = 13 ** 3 rows of the 7x7x7-window
# relative-position bias table.
_RELATIVE_POSITION_ROWS = 13**3


def load_network(weights_path: Path, *, merging_profile: MergingProfile = "monai-1.4") -> SwinUNETR:
    """Load converted safetensors into a ``SwinUNETR`` after checking every tensor.

    The names, shapes, dtypes and finiteness must match ``specification.json``;
    relative-position indices are stored as int32 (the source graph's int64) and
    range-checked. ``merging_profile`` selects the patch-merging graph (``"monai-1.4"``,
    the default and the released bundle's behavior, or ``"monai-1.6"``); the weights are
    the same.

    Raises:
        AssetNotReadyError: If the file is unreadable or its tensors differ from the
            pinned converted layout.
        MissingDependencyError: If the required MLX/Metal runtime cannot be admitted.
    """
    mx = import_mlx()
    parameters = specification()["parameters"]
    index_names = [name for name in parameters if name.endswith("relative_position_index")]
    weights = load_tensors(
        Path(weights_path),
        model_id=MODEL_ID,
        label="SwinUNETR",
        shapes={name: tuple(spec["shape"]) for name, spec in parameters.items()},
        dtypes={name: mx.int32 for name in index_names},
    )
    for name in index_names:
        if bool(mx.any((weights[name] < 0) | (weights[name] >= _RELATIVE_POSITION_ROWS)).item()):
            raise AssetNotReadyError(
                f"SwinUNETR weights {weights_path}: relative-position indices out of range: {name}",
                reason=f"mismatched {Path(weights_path).name}",
                hint="Download swinunetr.safetensors from MedMLX/swin-unetr-mlx on Hugging Face.",
            )
    return SwinUNETR(weights, mx=mx, merging_profile=merging_profile)
