"""Record pinned MONAI SwinUNETR logits on an HCC-TACE CT crop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from monai import __version__ as monai_version
from monai.bundle import ConfigParser

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_REVISION = "e3e994c2b1438f84f431162a6b319f2819500c65"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from real_inputs import network_input  # noqa: E402
from reference_data import seeded_native_weights  # noqa: E402


def main() -> None:
    """Record source logits and uncertainty-aware argmax cases, with no image in Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if monai_version != "1.4.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.4.0 and Torch 2.13 are required")
    if args.bundle.name != BUNDLE_REVISION:
        raise ValueError("Supply the pinned BTCV bundle snapshot")
    config = ConfigParser()
    config.read_config(str(args.bundle / "configs/inference.json"))
    model = config.get_parsed_content("network_def").cpu().eval()
    weights = seeded_native_weights(3)
    model.load_state_dict(
        {key: torch.from_numpy(value) for key, value in weights.items()}, strict=True
    )
    torch.set_num_threads(4)
    with torch.inference_mode():
        logits = model(torch.from_numpy(network_input().copy())).numpy()
    top = np.partition(logits, -2, axis=1)[:, -2:]
    margin = 6e-6 + 1e-5 * np.sum(np.abs(top), axis=1)
    ambiguous = np.flatnonzero((top[:, 1] - top[:, 0]) <= margin).astype(np.int32)
    scores = np.moveaxis(logits, 1, -1).reshape(-1, logits.shape[1])
    original = ROOT / "tests/fixtures/network.npz"
    with np.load(original, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
    metadata.pop("input_seed", None)
    metadata["logits_tolerance"] = {"atol": 3e-6, "rtol": 1e-5}
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    output = ROOT / "work/real-reference/network.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        metadata=json.dumps(metadata),
        sampled_logits=logits[:, :, ::8, ::8, ::8],
        argmax=logits.argmax(axis=1).astype(np.uint8),
        ambiguous_indices=ambiguous,
        ambiguous_logits=scores[ambiguous],
    )


if __name__ == "__main__":
    main()
