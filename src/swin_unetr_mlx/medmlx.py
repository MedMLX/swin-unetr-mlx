"""medmlx declaration and end-to-end runner for the SwinUNETR BTCV bundle.

The pinned upstream ``configs/inference.json`` defines preprocessing, sliding
windows, inversion and output layout. MLX supplies the network arithmetic;
``medmlx_core`` reads, admits and writes the NIfTI volumes; NumPy and SimpleITK
supply the volume transforms.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

from swin_unetr_mlx._arrays import Float32Array, Float64Array
from swin_unetr_mlx._records import RunResult, Runtime

if TYPE_CHECKING:
    from medmlx_core.volume_workspace import VolumeWorkspace

    from swin_unetr_mlx.network import SwinUNETR

import numpy as np
from medmlx_core.errors import InvalidInputError, ModelExecutionError
from medmlx_core.files import input_file, output_targets, publish
from medmlx_core.outputs import artifact
from medmlx_core.transforms import axis_codes, orient, scale_intensity_range
from medmlx_core.volumes import read_volume, write_volume
from medmlx_core.weights import weights_file, weights_revision

from swin_unetr_mlx import transforms
from swin_unetr_mlx.specification import bundle_config
from swin_unetr_mlx.weights import load_network

CONFIG = bundle_config()
# Source affine for ``Orientation.inverse``, resampling record, resampled affine.
type Inverse = tuple[Float64Array, transforms.Resampling, Float64Array]
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


#: The mask's file name inside ``output_dir``.
OUTPUT_NAME = "mask.nii.gz"


def run(
    *,
    image_path: str | Path,
    weights_path: str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
    sw_batch_size: int = CONFIG.sw_batch_size,
    streaming: bool = False,
    scratch_dir: str | Path | None = None,
) -> RunResult:
    """Segment abdominal organs in one CT volume and write the label map.

    ``image_path`` is expected to be a 3D CT NIfTI-1 or NIfTI-2 in Hounsfield units,
    admitted by ``medmlx_core.volumes.read_volume`` and readable by ITK. Omit
    ``weights_path`` to download and reuse ``MedMLX/swin-unetr-mlx``.
    An explicit model folder contains ``swinunetr.safetensors``.
    Writes ``mask.nii.gz`` in ``output_dir``, with the input shape and ITK reader's
    affine, and float32 labels 0-13 as ``SaveImaged`` does. An existing mask is
    replaced only when ``overwrite`` is true (default false).
    The image is rejected unless NiBabel's affine, converted to millimeters, matches
    the ITK affine, so conflicting qform/sform geometry never selects a reader's choice.

    Args:
        image_path: Finite 3D CT NIfTI in Hounsfield units, readable by NiBabel and ITK.
        weights_path: Optional folder containing ``swinunetr.safetensors``; omit to
            download and reuse the published model.
        output_dir: Directory for ``mask.nii.gz``, the restored organ label map.
        overwrite: Replace an existing mask when true; defaults to False.
        sw_batch_size: Positive number of windows evaluated together; source default four.
        streaming: Blend into temporary mapped logits instead of a resident volume.
        scratch_dir: Existing parent directory for streaming maps; system temporary
            storage is used when omitted. Maps are removed after restoration.

    Returns:
        ``{"outputs": {"mask": path}, "artifacts": {"mask": ...}, "details": {...}}``. The
        mask artifact is a ``label_map`` in ``image_physical`` space on ``image_path``'s grid,
        labelled with the bundle's 14 classes (0 background and 13 organs). ``details`` is
        described by ``swin_unetr_mlx._records.Details`` and always has
        ``requires_human_review`` true.

    Raises:
        InvalidInputError: ``overwrite`` is not a bool, the output exists without
            ``overwrite``, the image is rejected by ``read_volume``, or it reads with
            different shapes or affines through ITK and nibabel.
        AssetNotReadyError: ``weights_path`` is not a folder, or its
            ``swinunetr.safetensors`` is missing or does not match the pinned graph.
        ModelExecutionError: Inference failed or the mask is off the source grid or has
            labels outside 0-13.
        MissingDependencyError: The host is unsupported or the MLX Metal GPU is unavailable.
    """
    if type(sw_batch_size) is not int or sw_batch_size < 1:
        raise InvalidInputError("sw_batch_size must be a positive integer")
    if type(streaming) is not bool:
        raise InvalidInputError("streaming must be boolean")
    scratch = Path(scratch_dir).expanduser() if scratch_dir is not None else None
    if streaming and scratch is not None and not scratch.is_dir():
        raise InvalidInputError("scratch_dir must name an existing directory")
    image_file = input_file("image_path", image_path)
    mask_path = output_targets(
        output_dir, {"mask": OUTPUT_NAME}, overwrite=overwrite, inputs=[image_file]
    )["mask"]
    weights = weights_file("swin-unetr", weights_path, "swinunetr.safetensors")
    started = time.perf_counter()
    volume = read_volume(image_file)
    data = volume.data
    shape = (int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    try:
        reader_shape, reader_affine = transforms.read_itk_geometry(image_file)
    except RuntimeError as error:
        raise InvalidInputError(f"ITK cannot read image_path: {image_file}: {error}") from error
    if reader_shape != shape:
        raise InvalidInputError(
            f"image_path reads as {reader_shape} through ITK but {shape} through nibabel"
        )
    _validate_geometry(reader_affine, volume.affine, "image_path")
    network = load_network(weights)
    loaded = time.perf_counter()
    image, inverse = preprocess(data, reader_affine)
    prepared = time.perf_counter()
    if streaming:
        from medmlx_core.volume_workspace import VolumeWorkspace

        context = VolumeWorkspace(scratch)
    else:
        context = nullcontext(None)
    with context as workspace:
        logits, runtime = _predict(
            network, image, sw_batch_size=sw_batch_size, workspace=workspace,
        )
        predicted = time.perf_counter()
        # Reuse resident or mapped logits for probabilities through restoration.
        probabilities = transforms.softmax(logits[0], out=logits[0])
        del logits
        mask = _restore(probabilities, inverse)
        del probabilities
    _validate_mask(mask, shape)
    publish({
        mask_path: lambda partial: write_volume(
            partial, mask, reader_affine, dtype=np.dtype(CONFIG.output_dtype).type
        )
    })
    finished = time.perf_counter()
    return {
        "outputs": {"mask": mask_path.as_posix()},
        "artifacts": {
            "mask": artifact(
                kind="label_map",
                coordinate_space="image_physical",
                grid_of="image_path",
                labels=[{"value": value, "name": name} for value, name in LABELS.items()],
            )
        },
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
                "sw_batch_size": sw_batch_size,
                "overlap": CONFIG.overlap,
            },
            "streaming": streaming,
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
) -> tuple[Float32Array, Inverse]:
    """``EnsureChannelFirstd`` through ``EnsureTyped``: RAS, 1.5 x 1.5 x 2 mm, [0, 1].

    ``data`` is the float32 CT in HU, ``(X, Y, Z)`` as read by ``transforms.read_itk``,
    with its RAS ``affine``. It is reoriented to RAS, resampled to the bundle's spacing
    (mm), then clipped to -175..250 HU and scaled to [0, 1]. Returns the
    ``(1, 1, *spatial)`` float32 network input and the record ``Invertd`` reverses:
    the source affine ``Orientation`` restores, the resampling record and the resampled
    affine.
    """
    oriented, oriented_affine = orient(data[None], affine, CONFIG.axcodes)
    resampled, resampled_affine, resampling = transforms.spacing(
        oriented, oriented_affine, CONFIG.pixdim
    )
    image = scale_intensity_range(resampled, *CONFIG.intensity)
    return image[None], (np.array(affine, dtype=np.float64), resampling, resampled_affine)


def postprocess(
    logits: Float32Array,
    inverse: Inverse,
) -> Float32Array:
    """Softmax, ``Invertd`` (resampling then reorientation), argmax; float32 labels.

    ``logits`` is ``(14, *spatial)`` on the resampled RAS grid. As in the bundle, the
    softmax probabilities are resampled back to the source grid (bilinear) before the
    argmax. Returns the label map on the source grid.
    """
    return _restore(transforms.softmax(logits), inverse)


def _restore(
    probabilities: Float32Array,
    inverse: Inverse,
) -> Float32Array:
    """Resample, argmax, then reorient: ``postprocess`` after its softmax.

    Argmax runs per resampling chunk, and reorientation, a spatial flip/transpose
    that leaves the channel axis alone, then moves one label channel instead of 14.
    """
    source_affine, resampling, affine = inverse
    labels, restored = transforms.invert_spacing(
        probabilities, affine, resampling, channel_argmax=True
    )
    # ``Orientation.inverse`` reorients to the axis codes of the recorded source affine.
    labels, _ = orient(labels, restored, axis_codes(source_affine))
    return labels[0]


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
            f"{np.round(nibabel_affine, 4).tolist()}; "
            "set a qform or sform code with matching matrices"
        )


def _predict(
    network: SwinUNETR, image: Float32Array, *, sw_batch_size: int = CONFIG.sw_batch_size,
    workspace: VolumeWorkspace | None = None,
) -> tuple[Float32Array, Runtime]:
    """Run the bundle's windows with selected batching and optional mapped logits.

    ``image`` is the preprocessed ``(1, 1, *spatial)`` array. Returns float32 logits
    ``(1, 14, *spatial)`` and the MLX device and peak memory in bytes.
    """
    from medmlx_core.sliding_window import sliding_window_inference

    mx = network.mx
    mx.reset_peak_memory()
    try:
        logits = sliding_window_inference(
            np.ascontiguousarray(image, dtype=np.float32),
            CONFIG.roi_size,
            sw_batch_size,
            network,
            overlap=CONFIG.overlap,
            workspace=workspace,
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


__all__ = ["MODEL", "OUTPUT_NAME", "run"]
