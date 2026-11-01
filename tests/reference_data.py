"""Read fixed macOS arm64 upstream recordings without importing an oracle."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from fixture_contracts import Metadata

from swin_unetr_mlx._arrays import NumericArray
from swin_unetr_mlx.specification import specification

ROOT = Path(__file__).resolve().parent / "fixtures"


@dataclass(frozen=True, slots=True)
class ReferenceRecording:
    arrays: dict[str, NumericArray]
    metadata: Metadata

    def __post_init__(self) -> None:
        """Validate fixture schema, upstream revision, bundle, and host provenance."""
        spec = specification()
        if (
            self.metadata["schema_version"] != 2
            or self.metadata["recording_host"] != {"system": "Darwin", "machine": "arm64"}
            or self.metadata["upstream"] != spec["upstream"]
            or self.metadata["bundle_revision"] != spec["bundle"]["revision"]
        ):
            raise ValueError("Fixture provenance does not match the pinned upstream source")


def recording(name: str) -> ReferenceRecording:
    """Load a named reference archive and validate its recorded metadata."""
    path = ROOT / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing upstream recording: {path}; run make_medmlx_fixtures.py")
    with np.load(path, allow_pickle=False) as data:
        return ReferenceRecording(
            {key: data[key] for key in data.files if key != "metadata"},
            cast(Metadata, json.loads(str(data["metadata"]))),
        )


def seeded_native_weights(seed: int) -> dict[str, NumericArray]:
    """Reconstruct native graph tensors for the recorded seeded source; none are distributed."""
    rng = np.random.default_rng(seed)
    state: dict[str, NumericArray] = {}
    for name, parameter in specification()["parameters"].items():
        shape = parameter["shape"]
        # Draw in source order so the native tensors match the recorded seeded state.
        if name.endswith("relative_position_index"):
            state[name] = rng.integers(0, 13**3, shape).astype(np.int32)
        elif not name.endswith("num_batches_tracked"):
            state[name] = rng.normal(0.0, 0.02, shape).astype(np.float32)
    return state
