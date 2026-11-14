"""Public converted-weight inference against the original upstream graph."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX


import numpy as np
from fixture_contracts import NetworkMetadata
from medmlx_core.typing import ArrayFactory
from real_inputs import network_input
from reference_data import ReferenceRecording, recording, seeded_native_weights

from swin_unetr_mlx import forward_patch, load_network
from swin_unetr_mlx._arrays import Float32Array, Int32Array, UInt8Array


def _assert_network_reference(actual: Float32Array, reference: ReferenceRecording) -> None:
    """Compare recorded logits and label decisions within source-derived error bounds."""
    expected = cast(UInt8Array, reference.arrays["argmax"])
    assert actual.shape == (expected.shape[0], 14, *expected.shape[1:])
    assert actual.dtype == np.float32 and np.isfinite(actual).all()
    # Across the real CT crop, the maximum absolute source/Metal error was 2.921e-6.
    tolerance = {"atol": 3e-6, "rtol": 1e-5}
    assert cast(NetworkMetadata, reference.metadata)["logits_tolerance"] == tolerance
    stride = cast(NetworkMetadata, reference.metadata)["sample_stride"]
    # Torch and Metal differ in FP32 reduction order, hence a nonzero bound.
    np.testing.assert_allclose(
        actual[:, :, ::stride, ::stride, ::stride],
        reference.arrays["sampled_logits"],
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )
    indices = cast(Int32Array, reference.arrays["ambiguous_indices"])
    scores = np.moveaxis(actual, 1, -1).reshape(-1, actual.shape[1])
    # Argmax can change within this bound when the winner's lower error bound
    # overlaps another class's upper bound. Check all 14 logits at those voxels.
    np.testing.assert_allclose(
        scores[indices],
        reference.arrays["ambiguous_logits"],
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    )
    separated = np.ones(expected.size, dtype=bool)
    separated[indices] = False
    # A separated winner cannot change within the same per-logit error bound.
    np.testing.assert_array_equal(
        actual.argmax(axis=1).ravel()[separated],
        expected.ravel()[separated],
    )


def test_seeded_network_matches_upstream(tmp_path: Path, mx: MLX) -> None:
    """Compare seeded native network outputs with the pinned source recording."""
    reference = recording("network")
    meta = cast(NetworkMetadata, reference.metadata)
    weights = seeded_native_weights(meta["weights_seed"])
    path = tmp_path / "network.safetensors"
    mx.save_safetensors(
        str(path), {key: cast(ArrayFactory, mx.array)(value) for key, value in weights.items()}
    )
    network = load_network(path)
    values = network_input()
    _assert_network_reference(forward_patch(network, values), reference)
