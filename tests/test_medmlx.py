"""Bundle transform parity with MONAI, an end-to-end run and argument rejection.

The parity fixture is recorded by scripts/make_medmlx_fixtures.py, which runs the
pinned upstream MONAI 1.4 transforms on seeded inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fixture_contracts import RunnerMetadata
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.typing import ArrayFactory
from reference_data import recording, seeded_native_weights

from swin_unetr_mlx import medmlx, transforms
from swin_unetr_mlx._nifti import nib
from swin_unetr_mlx.medmlx import run

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX

CASES = ("axis", "flipped", "oblique")


def _image(tmp_path: Path, case: str) -> Path:
    """Write a recorded NIfTI input for the requested geometry."""
    path = tmp_path / f"{case}.nii.gz"
    path.write_bytes(recording("medmlx_btcv").arrays[f"{case}::image_nifti"].tobytes())
    return path


@pytest.mark.parametrize("case", CASES)
def test_transforms_match_the_bundle(tmp_path: Path, case: str) -> None:
    """Compare preprocessing and inverse transforms with recorded MONAI cases."""
    arrays = recording("medmlx_btcv").arrays
    data, affine = transforms.read_itk(_image(tmp_path, case))
    image, inverse = medmlx.preprocess(data, affine)
    assert image.dtype == np.float32
    if case == "axis":
        # CPU affine-grid rounding differs at 1/800 voxels, by 3.7253e-8
        # after HU scaling; 4e-8 bounds that observed upstream ARM difference.
        np.testing.assert_allclose(image[0], arrays[f"{case}::image"], rtol=0, atol=4e-8)
    else:
        np.testing.assert_array_equal(image[0], arrays[f"{case}::image"])
    logits = arrays[f"{case}::logits"].astype(np.float32)
    # Torch and NumPy exp/reductions differ by at most 4 FP32 ULPs here.
    np.testing.assert_array_max_ulp(
        transforms.softmax(logits), arrays[f"{case}::softmax"], maxulp=4
    )
    mask = medmlx.postprocess(logits, inverse)
    assert mask.dtype == np.float32
    assert np.array_equal(mask, arrays[f"{case}::mask"])


def test_run_matches_upstream(tmp_path: Path, mx: MLX) -> None:
    """One 96^3 window of the released graph with seeded weights matches the upstream mask."""
    source = tmp_path / "ct.nii.gz"
    reference = recording("runner")
    source.write_bytes(reference.arrays["image_nifti"].tobytes())
    saved = cast(RunnerMetadata, reference.metadata)["saved"]
    folder, blob = tmp_path / "model", tmp_path / "cached-blob"
    native = {k: cast(ArrayFactory, mx.array)(v) for k, v in seeded_native_weights(3).items()}
    mx.save_safetensors(str(tmp_path / "weights.safetensors"), native)
    # Match a Hugging Face snapshot link to an extensionless cache blob.
    (tmp_path / "weights.safetensors").rename(blob)
    folder.mkdir()
    (folder / "swinunetr.safetensors").symlink_to(blob)
    output_dir = tmp_path / "out"
    result = run(image_path=str(source), weights_path=str(folder), output_dir=str(output_dir))
    assert result["outputs"] == {"mask": (output_dir.resolve() / saved["path"]).as_posix()}
    written = nib.load(result["outputs"]["mask"])
    data = np.asanyarray(written.dataobj)
    assert np.array_equal(written.affine, np.array(saved["affine"]))
    assert data.dtype == np.float32
    assert np.array_equal(data, reference.arrays["mask"]), "Saved mask differs from upstream"


@pytest.mark.parametrize(
    ("problem", "error"),
    [("torch_weights", AssetNotReadyError), ("overwrite_not_bool", InvalidInputError),
     ("existing_output", InvalidInputError)],
)
def test_run_rejects_invalid_arguments_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str, error: type[Exception],
) -> None:
    """Reject each invalid argument category before any weights are downloaded."""

    def no_download(model_id: str) -> Path:
        """Fail the test if the runner reaches the weights download."""
        raise AssertionError(f"downloaded {model_id}")

    monkeypatch.setattr(medmlx, "weights_dir", no_download)
    image, output = _image(tmp_path, "axis"), tmp_path / "out"
    weights, overwrite = None, False
    if problem == "torch_weights":
        (tmp_path / "model.pt").write_bytes(b"unconverted")
        weights = str(tmp_path / "model.pt")
    elif problem == "overwrite_not_bool":
        overwrite = cast(bool, "true")
    else:
        (output / "axis").mkdir(parents=True)
        (output / "axis/axis_trans.nii.gz").write_bytes(b"existing output")
    with pytest.raises(error):
        run(
            image_path=str(image), weights_path=weights, output_dir=str(output),
            overwrite=overwrite,
        )
