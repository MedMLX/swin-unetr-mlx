"""medmlx declaration and end-to-end runner for the SwinUNETR BTCV bundle.

The pinned upstream ``configs/inference.json`` defines preprocessing, sliding
windows, inversion and output layout. MLX supplies the network arithmetic;
NumPy and SimpleITK supply the volume transforms.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

from swin_unetr_mlx._arrays import Float32Array, Float64Array
from swin_unetr_mlx._records import RunResult, Runtime

if TYPE_CHECKING:
    from swin_unetr_mlx._types import CoreSlidingWindow
    from swin_unetr_mlx.network import SwinUNETR

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.weights import weights_dir, weights_revision

from swin_unetr_mlx import transforms
from swin_unetr_mlx.specification import bundle_config

CONFIG = bundle_config()
BUNDLE = CONFIG.name
LABELS = dict(enumerate(CONFIG.labels))

MODEL = {
    "id": "swin-unetr",
    "label": "SwinUNETR BTCV",
    "task": "segmentation",
    "run": "swin_unetr_mlx.medmlx:run",
    # Measured MLX peak for a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 7.5,
    "summary": ("MONAI swin_unetr_btcv_segmentation: 13 abdominal organs in contrast-enhanced CT."),
    "labels": {str(value): name for value, name in LABELS.items()},
}


def run(
    *,
    image_path: str,
    weights_path: str | None = None,
    output_dir: str,
    overwrite: bool = False,
) -> RunResult:
    """Segment abdominal organs in one CT volume and write the label map.

    ``image_path`` is expected to be a 3D CT NIfTI in Hounsfield units; both
    NiBabel and ITK must read it, but the runner does not explicitly check its
    format. Omit ``weights_path`` to download and reuse ``MedMLX/swin-unetr-mlx``.
    An explicit model folder contains ``swinunetr.safetensors``. Native
    ``.safetensors`` checkpoint-file paths remain accepted.
    Writes ``<input>/<input>_trans.nii.gz`` under ``output_dir``, with the input
    shape and ITK reader's affine, and float32 labels 0-13 as ``SaveImaged`` does. An
    existing mask is replaced only when ``overwrite`` is true (default false).
    The image is rejected unless NiBabel's affine, converted to millimetres, matches
    the ITK affine, so conflicting qform/sform geometry never selects a reader's choice.

    Args:
        image_path: Finite 3D CT NIfTI in Hounsfield units, readable by NiBabel and ITK.
        weights_path: Optional folder containing ``swinunetr.safetensors``; omit to
            download and reuse the published model. Explicit files remain accepted.
        output_dir: Output root for the restored organ label map.
        overwrite: Replace an existing mask when true; defaults to False.

    Returns:
        ``{"outputs": {"mask": path}, "details": {...}}``; ``details`` is described by
        ``swin_unetr_mlx._records.Details`` and always has ``requires_human_review`` true.

    Raises:
        InvalidInputError: ``overwrite`` is not a bool, the output exists without
            ``overwrite``, or the image is missing, not a finite 3D NIfTI with positive
            spacing, or reads with different shapes or affines through ITK and nibabel.
        AssetNotReadyError: The weights file is missing, not ``.safetensors``, or does
            not match the pinned graph.
        ModelExecutionError: Inference failed or the mask is off the source grid or has
            labels outside 0-13.
        MissingDependencyError: The host is unsupported or the MLX Metal GPU is unavailable.
    """
    if type(overwrite) is not bool:
        raise InvalidInputError("overwrite must be boolean")
    image_file = Path(image_path).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    stem = Path(image_file.stem).stem if image_file.suffix == ".gz" else image_file.stem
    folder = output_root / stem if CONFIG.separate_folder else output_root
    mask_path = folder / f"{stem}_{CONFIG.output_postfix}.nii.gz"
    if mask_path.exists() and not overwrite:
        raise InvalidInputError(f"Output exists: {mask_path}; pass overwrite=true to replace it")
    bundle_dir = (
        weights_dir("swin-unetr") if weights_path is None else Path(weights_path).expanduser()
    )
    # Preserve explicit checkpoint-file callers and named cache symlinks.
    checkpoint = bundle_dir / "swinunetr.safetensors" if bundle_dir.is_dir() else bundle_dir
    weights = _weights_file(str(checkpoint))
    started = time.perf_counter()
    shape, nibabel_affine = _validate_volume(image_file, "image_path")
    try:
        data, reader_affine = transforms.read_itk(image_file)
    except RuntimeError as error:
        raise InvalidInputError(f"ITK cannot read image_path: {image_file}: {error}") from error
    if data.shape != shape:
        raise InvalidInputError(
            f"image_path reads as {data.shape} through ITK but {shape} through nibabel"
        )
    _validate_geometry(reader_affine, nibabel_affine, "image_path")
    network = _load_network(weights)
    loaded = time.perf_counter()
    image, inverse = preprocess(data, reader_affine)
    prepared = time.perf_counter()
    logits, runtime = _predict(network, image)
    predicted = time.perf_counter()
    # Softmax into the logits buffer and drop the logits, as ``postprocess`` would
    # otherwise keep both full-size arrays alive during resampling.
    probabilities = transforms.softmax(logits[0], out=logits[0])
    del logits
    mask = _restore(probabilities, inverse)
    del probabilities
    _validate_mask(mask, shape)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(mask_path, mask, reader_affine)
    finished = time.perf_counter()
    return {
        "outputs": {"mask": mask_path.as_posix()},
        "details": {
            "weights_revision": weights_revision(weights.parent),
            "bundle": BUNDLE,
            "labels": {str(value): name for value, name in LABELS.items()},
            "labels_present": sorted(int(value) for value in np.unique(mask)),
            "input_shape": list(shape),
            "network_input_shape": list(image.shape[2:]),
            "network_spacing_mm": list(CONFIG.pixdim),
            "output_dtype": CONFIG.output_dtype,
            "sliding_window": {
                "roi_size": list(CONFIG.roi_size),
                "sw_batch_size": CONFIG.sw_batch_size,
                "overlap": CONFIG.overlap,
            },
            "mlx_device": runtime["device"],
            "peak_mlx_bytes": runtime["peak_bytes"],
            "seconds": {
                "load_inputs_and_weights": round(loaded - started, 4),
                "preprocess": round(prepared - loaded, 4),
                "inference": round(predicted - prepared, 4),
                "postprocess_and_write": round(finished - predicted, 4),
            },
            "requires_human_review": True,
        },
    }


def preprocess(
    data: Float32Array,
    affine: Float64Array,
) -> tuple[Float32Array, tuple[transforms.Orientation, transforms.Resampling, Float64Array]]:
    """``EnsureChannelFirstd`` through ``EnsureTyped``: RAS, 1.5 x 1.5 x 2 mm, [0, 1].

    ``data`` is the float32 CT in HU, ``(X, Y, Z)`` as read by ``transforms.read_itk``,
    with its RAS ``affine``. It is reoriented to RAS, resampled to the bundle's spacing
    (mm), then clipped to -175..250 HU and scaled to [0, 1]. Returns the
    ``(1, 1, *spatial)`` float32 network input and the record ``Invertd`` reverses.
    """
    oriented, oriented_affine, orientation = transforms.orient(data[None], affine, CONFIG.axcodes)
    resampled, resampled_affine, resampling = transforms.spacing(
        oriented, oriented_affine, CONFIG.pixdim
    )
    image = transforms.scale_intensity_range(resampled, *CONFIG.intensity)
    return image[None], (orientation, resampling, resampled_affine)


def postprocess(
    logits: Float32Array,
    inverse: tuple[transforms.Orientation, transforms.Resampling, Float64Array],
) -> Float32Array:
    """Softmax, ``Invertd`` (resampling then reorientation), argmax; float32 labels.

    ``logits`` is ``(14, *spatial)`` on the resampled RAS grid. As in the bundle, the
    softmax probabilities are resampled back to the source grid (bilinear) before the
    argmax. Returns the label map on the source grid.
    """
    return _restore(transforms.softmax(logits), inverse)


def _restore(
    probabilities: Float32Array,
    inverse: tuple[transforms.Orientation, transforms.Resampling, Float64Array],
) -> Float32Array:
    """Resample, argmax, then reorient: ``postprocess`` after its softmax.

    Argmax runs per resampling chunk, and reorientation, a spatial flip/transpose
    that leaves the channel axis alone, then moves one label channel instead of 14.
    """
    orientation, resampling, affine = inverse
    labels, restored = transforms.invert_spacing(
        probabilities, affine, resampling, channel_argmax=True
    )
    labels, _ = transforms.invert_orientation(labels, restored, orientation)
    return labels[0]


def _weights_file(weights_path: str) -> Path:
    """Require an existing ``.safetensors`` file, keeping symlinked cache names intact."""
    # Preserve the named Hub-cache symlink: its underlying blob has no extension,
    # while MLX determines the tensor format from the supplied filename.
    path = Path(weights_path).expanduser().absolute()
    hint = (
        "Download swinunetr.safetensors from MedMLX/swin-unetr-mlx on Hugging Face."
    )
    if not path.is_file():
        raise AssetNotReadyError(
            f"SwinUNETR weights not found: {path}", reason=f"missing {path}", hint=hint
        )
    if path.suffix != ".safetensors":
        raise AssetNotReadyError(
            f"SwinUNETR weights must be converted MLX safetensors, got {path.name}",
            reason=f"unconverted {path.name}",
            hint=hint,
        )
    return path


def _validate_volume(path: Path, role: str) -> tuple[tuple[int, int, int], Float64Array]:
    """Admit a real, finite float32-range 3D NIfTI with valid spatial units.

    Returns the shape and NiBabel's RAS affine converted to millimetres, the unit
    ITK reports: metres and microns are scaled and unspecified units count as mm.
    """
    from swin_unetr_mlx._nifti import ImageFileError, nib

    if not path.is_file():
        raise InvalidInputError(f"{role} not found: {path}")
    try:
        image = nib.load(path.as_posix())
    except (OSError, ImageFileError) as error:
        raise InvalidInputError(f"Invalid {role} NIfTI: {path}") from error
    # NIfTI stores the spatial unit in the low three bits; SimpleITK otherwise
    # silently treats unrecognized unit codes as millimetres.
    if (int(image.header["xyzt_units"]) & 0x07) not in (0, 1, 2, 3):
        raise InvalidInputError(f"{role} has invalid NIfTI spatial units")
    raw = np.asanyarray(image.dataobj)
    if raw.ndim != 3:
        raise InvalidInputError(f"{role} must be a 3D NIfTI; got shape {tuple(raw.shape)}")
    if np.iscomplexobj(raw):
        raise InvalidInputError(f"{role} must contain real-valued voxels")
    if not np.all(np.isfinite(raw)):
        raise InvalidInputError(f"{role} contains non-finite voxels")
    limit = np.finfo(np.float32).max
    if np.any(raw > limit) or np.any(raw < -limit):
        raise InvalidInputError(f"{role} contains voxels outside the finite float32 range")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
        raise InvalidInputError(f"{role} affine must be a finite 4x4 matrix")
    zooms = np.asarray(image.header.get_zooms()[:3], dtype=np.float64)
    if not np.all(np.isfinite(zooms)) or np.any(zooms <= 0.0):
        raise InvalidInputError(f"{role} voxel spacing values must be finite and positive")
    # NIfTI unit codes 1-3 are metre, millimetre, micron; 0 (unknown) reads as mm.
    millimetres = {0: 1.0, 1: 1000.0, 2: 1.0, 3: 0.001}[int(image.header["xyzt_units"]) & 0x07]
    affine = affine * np.array([[millimetres], [millimetres], [millimetres], [1.0]])
    return (int(raw.shape[0]), int(raw.shape[1]), int(raw.shape[2])), affine


def _validate_geometry(itk_affine: Float64Array, nibabel_affine: Float64Array, role: str) -> None:
    """Reject a NIfTI whose physical geometry differs between ITK and NiBabel.

    The readers resolve qform and sform differently: NiBabel prefers any sform,
    ITK prefers a qform over a non-scanner sform or a sheared one, and with neither
    form code they assume different origins. Agreement within 1e-3 mm (the stored
    forms are float32) means the voxel-to-RAS mapping used for preprocessing and the
    written mask is the one both readers report.
    """
    if not np.allclose(itk_affine, nibabel_affine, rtol=1e-6, atol=1e-3):
        raise InvalidInputError(
            f"{role} has conflicting NIfTI geometry: ITK affine "
            f"{np.round(itk_affine, 4).tolist()} differs from NiBabel affine "
            f"{np.round(nibabel_affine, 4).tolist()}; set a qform or sform code with matching matrices"
        )


def _load_network(weights: Path) -> SwinUNETR:
    """Load the checkpoint, reporting any layout mismatch as ``AssetNotReadyError``."""

    from swin_unetr_mlx.weights import load_model

    try:
        return load_model(weights)
    except (InvalidInputError, ValueError, RuntimeError) as error:
        raise AssetNotReadyError(
            f"SwinUNETR weights {weights} do not match the pinned graph: {error}",
            reason=f"mismatched {weights.name}",
            hint="Download swinunetr.safetensors from MedMLX/swin-unetr-mlx on Hugging Face.",
        ) from error


def _predict(network: SwinUNETR, image: Float32Array) -> tuple[Float32Array, Runtime]:
    """The bundle's ``SlidingWindowInferer`` (96^3, batch 4, overlap 0.5, constant).

    ``image`` is the preprocessed ``(1, 1, *spatial)`` array. Returns float32 logits
    ``(1, 14, *spatial)`` and the MLX device and peak memory in bytes.
    """
    import medmlx_core.sliding_window as core_window

    sliding_window_inference = cast("CoreSlidingWindow", core_window).sliding_window_inference

    mx = network.mx
    mx.reset_peak_memory()
    try:
        logits = sliding_window_inference(
            np.ascontiguousarray(image, dtype=np.float32),
            CONFIG.roi_size,
            CONFIG.sw_batch_size,
            network,
            overlap=CONFIG.overlap,
        )
    except (RuntimeError, ValueError) as error:
        raise ModelExecutionError(f"SwinUNETR MLX inference failed: {error}") from error
    if not np.isfinite(logits).all():
        raise ModelExecutionError("SwinUNETR produced non-finite logits")
    return logits, {"device": str(mx.default_device()), "peak_bytes": mx.get_peak_memory()}


def _validate_mask(mask: Float32Array, shape: tuple[int, int, int]) -> None:
    """Check the bundle's source-grid and label contracts before publishing a mask."""
    if tuple(mask.shape) != shape:
        raise ModelExecutionError(f"SwinUNETR mask shape {mask.shape} is not the input {shape}")
    if not set(np.unique(mask).tolist()) <= set(LABELS):
        raise ModelExecutionError("SwinUNETR exported labels outside the bundle contract")


def write_atomic(path: Path, mask: Float32Array, affine: Float64Array) -> None:
    """Write the mask NIfTI via a temporary file in ``path.parent``, then rename it.

    A failure therefore never leaves a partial file at ``path``.
    """
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".nii.gz", dir=path.parent)
    os.close(descriptor)
    try:
        transforms.write_nifti(Path(name), mask, affine, CONFIG.output_dtype)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


__all__ = ["MODEL", "run"]
