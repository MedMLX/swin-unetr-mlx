"""Released-source parity and strict converted-checkpoint admission."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX, Array
from pathlib import Path

import numpy as np
import pytest
from medmlx_core.errors import InvalidInputError
from medmlx_core.typing import ArrayFactory
from reference_data import seeded_native_weights

from swin_unetr_mlx import forward_patch, load_model


def test_converted_model_matches_independent_source(
    request: pytest.FixtureRequest
) -> None:
    """Compare an explicitly supplied checkpoint with independent source logits."""
    checkpoint = os.environ.get("MODEL_WEIGHTS")
    reference = os.environ.get("MODEL_SOURCE_OUTPUT")
    input_path = os.environ.get("MODEL_NETWORK_INPUT")
    if checkpoint is None or reference is None or input_path is None:
        pytest.skip("Set MODEL_WEIGHTS, MODEL_NETWORK_INPUT and MODEL_SOURCE_OUTPUT")
    request.getfixturevalue("mx")
    network = load_model(Path(checkpoint))
    values = np.load(input_path, allow_pickle=False)
    assert values.shape == (1, 1, 96, 96, 96)
    actual = forward_patch(network, values)
    expected = np.load(Path(reference) / "logits.npy", allow_pickle=False)
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
    np.testing.assert_array_equal(actual.argmax(axis=1), expected.argmax(axis=1))


@pytest.fixture(scope="module")
def converted_weights(
    tmp_path_factory: pytest.TempPathFactory,
    mx: MLX,
) -> dict[str, Array]:
    """Create and validate a seeded native checkpoint for admission cases."""
    destination = tmp_path_factory.mktemp("converted") / "model.safetensors"
    arrays = seeded_native_weights(3)
    mx.save_safetensors(
        str(destination),
        {name: cast(ArrayFactory, mx.array)(value) for name, value in arrays.items()},
    )
    load_model(destination)
    return cast("dict[str, Array]", mx.load(str(destination)))


@pytest.mark.parametrize("corruption", ["missing", "shape", "nonfinite", "index_range"])
def test_converted_checkpoint_admission(
    tmp_path: Path, converted_weights: dict[str, Array], corruption: str, mx: MLX
) -> None:
    """Reject missing, malformed, non-finite, or out-of-table checkpoint tensors."""
    weights = dict(converted_weights)
    name = next(key for key, value in weights.items() if value.dtype == mx.float32)
    value = weights[name]
    if corruption == "missing":
        del weights[name]
    elif corruption == "shape":
        weights[name] = mx.zeros((1,))
    elif corruption == "nonfinite":
        weights[name] = mx.full(value.shape, float("nan"))
    else:
        name = next(key for key in weights if key.endswith("relative_position_index"))
        weights[name] = mx.full(weights[name].shape, 13**3, dtype=mx.int32)
    destination = tmp_path / "invalid.safetensors"
    mx.save_safetensors(str(destination), weights)
    with pytest.raises(InvalidInputError):
        load_model(destination)
