"""Strict standalone safetensors loading."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
from medmlx_core.checkpoints import tensor_mapping_from_payload
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory

from swin_unetr_mlx._arrays import HostArray, NumericArray
from swin_unetr_mlx.network import MergingProfile, SwinUNETR
from swin_unetr_mlx.runtime import import_mlx
from swin_unetr_mlx.specification import specification

MODEL_ID = "swin_unetr_btcv_segmentation"


def _validate(payload: Mapping[str, object] | Mapping[object, object]) -> dict[str, HostArray]:
    """Check names, shapes, dtypes and finiteness against ``specification.json``.

    ``num_batches_tracked`` buffers are not expected and are dropped. Relative-position
    indices are stored as int32 and range-checked; other tensors keep the specified dtype.
    Returns C-contiguous host arrays.
    """
    expected = specification()["parameters"]
    expected = {
        key: value for key, value in expected.items() if not key.endswith("num_batches_tracked")
    }
    if set(payload) != set(expected):
        raise InvalidInputError(f"{MODEL_ID} checkpoint names differ from the pinned source graph")
    state = tensor_mapping_from_payload(payload, what="swin_unetr_btcv_segmentation")
    if set(state) != set(expected):
        raise InvalidInputError(f"{MODEL_ID} checkpoint values must all be tensors")
    arrays: dict[str, HostArray] = {}
    for name, spec in expected.items():
        value = cast(NumericArray, state[name])
        dtype = "int32" if name.endswith("relative_position_index") else spec["dtype"]
        if (
            list(value.shape) != spec["shape"]
            or str(value.dtype) != dtype
            or not np.isfinite(value).all()
        ):
            raise InvalidInputError(f"Invalid {MODEL_ID} checkpoint parameter: {name}")
        if name.endswith("relative_position_index"):
            # Indices address the (2 * 7 - 1) ** 3 = 13 ** 3 rows of the 7x7x7-window
            # relative-position bias table.
            if value.min() < 0 or value.max() >= 13**3:
                raise InvalidInputError(f"Invalid Swin relative-position indices: {name}")
            value = value.astype(np.int32)
        if not name.endswith("num_batches_tracked"):
            arrays[name] = np.ascontiguousarray(value)
    return arrays


def load_model(weights_path: Path, *, merging_profile: MergingProfile = "monai-1.4") -> SwinUNETR:
    """Load converted safetensors into a ``SwinUNETR`` after checking every tensor.

    ``merging_profile`` selects the patch-merging graph (``"monai-1.4"``, the default and
    the released bundle's behavior, or ``"monai-1.6"``); the weights are the same.

    Raises:
        InvalidInputError: If the loaded mapping or its tensors differ from the
            pinned converted layout.
        RuntimeError: If MLX cannot open or decode the weights file.
        MissingDependencyError: If the required MLX/Metal runtime cannot be admitted.
    """
    if Path(weights_path).suffix != ".safetensors":
        raise InvalidInputError("Weights must be a safetensors file")
    mx = import_mlx()
    payload = mx.load(str(weights_path))
    if not isinstance(payload, dict):
        raise InvalidInputError("Converted checkpoint must be a safetensors mapping")
    loaded = payload
    arrays = _validate(
        {name: cast(HostArray, np.asarray(value)) for name, value in loaded.items()}
    )
    weights = {name: cast(ArrayFactory, mx.array)(value) for name, value in arrays.items()}
    mx.eval(weights)
    return SwinUNETR(weights, mx=mx, merging_profile=merging_profile)
