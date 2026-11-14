"""Bundle transform parity with MONAI, an end-to-end run and argument rejection.

The parity fixture records pinned upstream MONAI 1.4 transforms on HCC-TACE CT.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fixture_contracts import RunnerMetadata
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.outputs import validate_outputs
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


@pytest.mark.parametrize("batch_size,streaming", [(4, False), (1, True), (2, True)])
def test_run_matches_upstream(tmp_path: Path, mx: MLX, batch_size: int, streaming: bool) -> None:
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
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = run(
        image_path=str(source), weights_path=str(folder), output_dir=str(output_dir),
        sw_batch_size=batch_size, streaming=streaming, scratch_dir=str(scratch),
    )
    assert result["details"]["sliding_window"]["sw_batch_size"] == batch_size
    assert result["details"]["streaming"] is streaming
    assert not list(scratch.iterdir())
    assert result["outputs"] == {"mask": (output_dir.resolve() / saved["path"]).as_posix()}
    mask_artifact = validate_outputs(result).artifacts["mask"]
    assert (mask_artifact["kind"], mask_artifact.get("grid_of")) == ("label_map", "image_path")
    assert mask_artifact.get("labels") == [
        {"value": value, "name": name} for value, name in medmlx.LABELS.items()
    ]
    written = nib.load(result["outputs"]["mask"])
    data = np.asanyarray(written.dataobj)
    assert np.array_equal(written.affine, np.array(saved["affine"]))
    assert data.dtype == np.float32
    assert np.array_equal(data, reference.arrays["mask"]), "Saved mask differs from upstream"


@pytest.mark.parametrize(
    ("problem", "error"),
    [("torch_weights", AssetNotReadyError), ("overwrite_not_bool", InvalidInputError),
     ("existing_output", InvalidInputError), ("missing_image", InvalidInputError)],
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
    elif problem == "missing_image":
        image = tmp_path / "absent.nii.gz"
    else:
        (output / "axis").mkdir(parents=True)
        (output / "axis/axis_trans.nii.gz").write_bytes(b"existing output")
    with pytest.raises(error):
        run(
            image_path=str(image), weights_path=weights, output_dir=str(output),
            overwrite=overwrite,
        )


def _affine(spacing: tuple[float, float, float], origin: tuple[float, float, float]) -> np.ndarray:
    """A non-identity direction (rotations about z then x) with the given spacing and origin."""
    c, s, cx, sx = np.cos(0.3), np.sin(0.3), np.cos(0.2), np.sin(0.2)
    rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag(spacing)
    affine[:3, 3] = origin
    return affine


ORIENTED = _affine((0.8, 0.9, 2.5), (-120.5, 80.25, 300.0))
AXIAL = np.diag([-0.8, 0.9, 2.5, 1.0])
METRES = ORIENTED * np.r_[1e-3, 1e-3, 1e-3, 1.0][:, None]

# (sform, sform_code, qform, qform_code, units, millimetres per unit, accepted)
GEOMETRIES = {
    "matching_forms": (ORIENTED, 1, ORIENTED, 1, "mm", 1.0, True),
    "qform_only": (ORIENTED, 0, ORIENTED, 1, "mm", 1.0, True),
    "metres": (METRES, 1, METRES, 1, "meter", 1000.0, True),
    # ITK takes the qform over an aligned (code 2) sform; NiBabel takes the sform.
    "conflicting_aligned_sform": (ORIENTED, 2, AXIAL, 1, "mm", 1.0, False),
    # ITK replaces a sheared sform with the qform.
    "sheared_sform": (ORIENTED + np.eye(4, k=1) * [0.3], 1, ORIENTED, 1, "mm", 1.0, False),
    # Without form codes the readers place the origin differently.
    "no_form_codes": (ORIENTED, 0, ORIENTED, 0, "mm", 1.0, False),
}


@pytest.mark.parametrize("case", GEOMETRIES)
def test_run_requires_reader_geometry_agreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Reject NIfTI forms that ITK and NiBabel map to different physical coordinates.

    Accepted files then fail only on the placeholder weights, and the ITK affine the
    runner writes equals NiBabel's affine in millimetres.
    """
    sform, sform_code, qform, qform_code, units, millimetres, accepted = GEOMETRIES[case]
    data = np.asanyarray(nib.load(_image(tmp_path, "axis")).dataobj)
    image = nib.Nifti1Image(data, sform)
    image.header.set_xyzt_units(units)
    image.set_sform(sform, sform_code)
    image.set_qform(qform, qform_code)
    path = tmp_path / f"{case}.nii.gz"
    nib.save(image, path)
    weights = tmp_path / "model" / "swinunetr.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(medmlx, "_load_network", _reject_weights)
    error = AssetNotReadyError if accepted else InvalidInputError
    with pytest.raises(error) as raised:
        run(image_path=str(path), weights_path=str(weights.parent), output_dir=str(tmp_path))
    if not accepted:
        assert "conflicting NIfTI geometry" in str(raised.value)
        return
    reference = nib.load(path).affine * np.r_[[millimetres] * 3, 1.0][:, None]
    np.testing.assert_allclose(transforms.read_itk(path)[1], reference, rtol=1e-6, atol=1e-3)
    header_shape, header_affine = transforms.read_itk_geometry(path)
    assert header_shape == data.shape
    np.testing.assert_allclose(header_affine, reference, rtol=1e-6, atol=1e-3)


def _reject_weights(weights: Path) -> None:
    """Stand in for the weights load reached only after the geometry check passes."""
    raise AssetNotReadyError("placeholder weights", reason="placeholder", hint="")


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, np.finfo(np.float64).max])
def test_runner_rejects_invalid_source_values_before_any_reader_cast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_value: float,
) -> None:
    source = tmp_path / "source.nii.gz"
    data = np.zeros((4, 5, 6), dtype=np.float64)
    data[1, 2, 3] = bad_value
    nib.save(nib.Nifti1Image(data, np.eye(4)), source)
    weights = tmp_path / "swinunetr.safetensors"
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(medmlx, "_load_network", _reject_weights)
    with pytest.raises(InvalidInputError, match=r"non-finite|finite float32 range"):
        run(image_path=str(source), weights_path=str(weights), output_dir=str(tmp_path / "out"))
