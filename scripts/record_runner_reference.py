"""Record MONAI BTCV transform and runner outputs on HCC-TACE CT."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai import __version__ as monai_version
from monai.bundle import ConfigParser
from monai.data import MetaTensor
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureTyped,
    Invertd,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

ROOT = Path(__file__).resolve().parents[1]
REVISION = "e3e994c2b1438f84f431162a6b319f2819500c65"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from reference_data import seeded_native_weights  # noqa: E402


def load_sitk(path: Path) -> MetaTensor:
    """Read source voxels and convert SimpleITK's LPS geometry to RAS."""
    image = sitk.ReadImage(str(path))
    voxels = sitk.GetArrayFromImage(image).T.astype(np.float32)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = np.asarray(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
    affine[:3, 3] = image.GetOrigin()
    affine[:2] *= -1
    return MetaTensor(torch.from_numpy(voxels[None].copy()), affine=affine)


def pipeline():
    """Build the pinned bundle transforms after its SimpleITK image read."""
    return Compose(
        [
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
            ScaleIntensityRanged(keys="image", a_min=-175, a_max=250, b_min=0, b_max=1, clip=True),
            EnsureTyped(keys="image"),
        ]
    )


def main() -> None:
    """Prepare real crops with controlled geometry and execute the source graph."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if monai_version != "1.4.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.4.0 and Torch 2.13 are required")
    if args.bundle.name != REVISION:
        raise ValueError("Supply the pinned BTCV bundle snapshot")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )
    out = ROOT / "work/real-reference"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(ROOT / "tests/fixtures/medmlx_btcv.npz", allow_pickle=False) as fixture:
        arrays = {key: fixture[key] for key in fixture.files}
    metadata = json.loads(arrays["metadata"].item())
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    pre = pipeline()
    with tempfile.TemporaryDirectory(prefix="swin-btcv-") as temp:
        temp_path = Path(temp)
        for case in metadata["cases"]:
            name, geometry = case["case"], case["saved"]
            shape = tuple(geometry["shape"])
            slices = tuple(
                slice(start, start + size)
                for start, size in zip((100, 210, 55), shape, strict=True)
            )
            path = temp_path / f"{name}.nii.gz"
            nib.save(nib.Nifti1Image(ct[slices], np.asarray(geometry["affine"])), path)
            arrays[f"{name}::image_nifti"] = np.frombuffer(path.read_bytes(), dtype=np.uint8)
            transformed = pre({"image": load_sitk(path)})
            arrays[f"{name}::image"] = transformed["image"].as_tensor().numpy()
        np.savez_compressed(out / "medmlx_btcv.npz", **arrays)
        with np.load(ROOT / "tests/fixtures/runner.npz", allow_pickle=False) as original:
            runner_metadata = json.loads(original["metadata"].item())
        runner_metadata.pop("input_seed", None)
        runner_metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
        saved = runner_metadata["saved"]
        shape = tuple(saved["shape"])
        slices = tuple(
            slice(start, start + size) for start, size in zip((100, 210, 55), shape, strict=True)
        )
        path = temp_path / "ct.nii.gz"
        nib.save(nib.Nifti1Image(ct[slices], np.asarray(saved["affine"])), path)
        image_bytes = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        data = pre({"image": load_sitk(path)})
        config = ConfigParser()
        config.read_config(str(args.bundle / "configs/inference.json"))
        model = config.get_parsed_content("network_def").cpu().eval()
        model.load_state_dict(
            {key: torch.from_numpy(value) for key, value in seeded_native_weights(3).items()},
            strict=True,
        )
        inferer = config.get_parsed_content("inferer")
        torch.set_num_threads(4)
        with torch.inference_mode():
            prediction = inferer(data["image"][None], model)[0]
        data["pred"] = MetaTensor(prediction)
        post = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                Invertd(
                    keys="pred",
                    transform=pre,
                    orig_keys="image",
                    nearest_interp=False,
                    to_tensor=True,
                ),
                AsDiscreted(keys="pred", argmax=True),
            ]
        )
        mask = post(data)["pred"].as_tensor().numpy()[0].astype(np.float32)
        np.savez_compressed(
            out / "runner.npz",
            metadata=json.dumps(runner_metadata),
            image_nifti=image_bytes,
            image=data["image"].as_tensor().numpy(),
            mask=mask,
        )


if __name__ == "__main__":
    main()
