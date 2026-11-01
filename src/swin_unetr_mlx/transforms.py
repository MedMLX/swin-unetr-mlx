"""NumPy ports of the MONAI 1.4 transforms in the pinned SwinUNETR BTCV bundle.

The bundle's ``configs/inference.json`` (revision e3e994c, pinned in
``specification.json``) reads CT with ``ITKReader``, reorients to RAS, resamples to
1.5 x 1.5 x 2.0 mm, scales -175..250 HU to [0, 1], and after the network applies
softmax, inverts the resampling and reorientation, and takes the argmax. These
functions reproduce those transforms without Torch or MONAI. ITK's NIfTI reader is
used through SimpleITK. Spatial resampling follows MONAI's ``AffineTransform`` path,
``torch.nn.functional.affine_grid`` and trilinear ``grid_sample`` with border
padding and ``align_corners=False``, computed in float64 as ``Spacing`` does and
returned as float32.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from medmlx_core.geometry import (
    _bmm as _bmm,
)
from medmlx_core.geometry import (
    _fma as _fma,
)
from medmlx_core.geometry import (
    _linspace_from_neg_one as _linspace_from_neg_one,
)
from medmlx_core.geometry import (
    _mm as _mm,
)
from medmlx_core.geometry import (
    affine_to_spacing as affine_to_spacing,
)
from medmlx_core.geometry import (
    compute_shape_offset as compute_shape_offset,
)
from medmlx_core.geometry import (
    to_affine_nd as to_affine_nd,
)
from medmlx_core.geometry import (
    zoom_affine as zoom_affine,
)

from swin_unetr_mlx._arrays import HostArray, Int64Array
from swin_unetr_mlx._nifti import nib

type Float32Array = npt.NDArray[np.float32]
type Float64Array = npt.NDArray[np.float64]

# monai.data.utils.AFFINE_TOL
AFFINE_TOL = 1e-3
_LABELS = (("L", "R"), ("P", "A"), ("I", "S"))
# Bound the float64 corner arrays of one resampling chunk.
_CHUNK_VOXELS = 1 << 20
# Resampling chunks run concurrently; each holds about ten chunk-sized arrays.
_RESAMPLE_THREADS = min(8, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class Orientation:
    """What ``Orientation`` records for its inverse: the affine it started from."""

    original_affine: Float64Array


@dataclass(frozen=True, slots=True)
class Resampling:
    """What ``SpatialResample`` records for its inverse."""

    src_affine: Float64Array
    orig_size: tuple[int, ...]


def read_itk(path: Path) -> tuple[Float32Array, Float64Array]:
    """``LoadImage(reader="ITKReader")``: float32 voxels in (x, y, z) order and the RAS affine.

    As MONAI's ``ITKReader``: the ITK array is transposed from (z, y, x), and the
    LPS affine ``direction @ diag(spacing)`` with the origin is flipped to RAS.
    """
    from swin_unetr_mlx._sitk import sitk

    image = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(image).T.astype(np.float32, copy=False)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    affine = np.eye(4)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, -1] = origin
    return data, np.diag([-1, -1, 1, 1]).astype(affine.dtype) @ affine










def orient(
    data: Float32Array,
    affine: Float64Array,
    axcodes: str,
) -> tuple[Float32Array, Float64Array, Orientation]:
    """``Orientation(axcodes)`` on channel-first data; returns data, new affine, inverse."""
    rank = data.ndim - 1
    affine_np = np.array(affine, dtype=np.float64, copy=True)
    affine_ = to_affine_nd(rank, affine_np)
    source = nib.io_orientation(affine_)
    target = nib.orientations.axcodes2ornt(axcodes[:rank], labels=_LABELS)
    spatial_ornt = nib.orientations.ornt_transform(source, target)
    xform = nib.orientations.inv_ornt_aff(spatial_ornt, data.shape[1:])
    spatial_ornt[:, 0] += 1
    spatial_ornt = np.concatenate([np.array([[0, 1]]), spatial_ornt])
    axes = [axis for axis, flip in enumerate(spatial_ornt[:, 1]) if flip == -1]
    full_transpose = np.arange(data.ndim)
    full_transpose[: len(spatial_ornt)] = np.argsort(spatial_ornt[:, 0])
    result = np.flip(data, tuple(axes)) if axes else data
    if not np.all(full_transpose == np.arange(data.ndim)):
        result = result.transpose(full_transpose)
    new_affine = _mm(affine_np, to_affine_nd(len(affine_np) - 1, xform))
    return np.ascontiguousarray(result), new_affine, Orientation(affine_np)


def invert_orientation(
    data: Float32Array,
    affine: Float64Array,
    record: Orientation,
) -> tuple[Float32Array, Float64Array]:
    """``Orientation.inverse``: reorient to the recorded original axis codes."""
    axcodes = "".join(
        cast(tuple[str, ...], nib.orientations.aff2axcodes(record.original_affine, labels=_LABELS))
    )
    result, new_affine, _ = orient(data, affine, axcodes)
    return result, new_affine


def spacing(
    data: Float32Array,
    affine: Float64Array,
    pixdim: Sequence[float],
    output_spatial_shape: Sequence[int] | None = None,
) -> tuple[Float32Array, Float64Array, Resampling]:
    """``Spacing(pixdim, mode="bilinear")`` with MONAI defaults (border, float64)."""
    rank = data.ndim - 1
    affine_ = to_affine_nd(rank, affine)
    out_d = np.array(tuple(pixdim), dtype=np.float64)[:rank].copy()
    if out_d.size < rank:
        out_d = np.append(out_d, [out_d[-1]] * (rank - out_d.size))
    original = affine_to_spacing(affine_, rank)
    for index, current in enumerate(original):
        target = out_d[index]
        keep = (target - AFFINE_TOL) <= current <= (target + AFFINE_TOL)
        out_d[index] = current if keep else target
    new_affine = zoom_affine(affine_, out_d)
    output_shape, offset = compute_shape_offset(data.shape[1:], affine_, new_affine)
    new_affine[:rank, -1] = offset[:rank]
    shape = list(output_shape) if output_spatial_shape is None else list(output_spatial_shape)
    return spatial_resample(data, affine, new_affine, shape)


def invert_spacing(
    data: Float32Array,
    affine: Float64Array,
    record: Resampling,
) -> tuple[Float32Array, Float64Array]:
    """``SpatialResample.inverse``: resample back to the recorded affine and size."""
    result, new_affine, _ = spatial_resample(data, affine, record.src_affine, record.orig_size)
    return result, new_affine


def spatial_resample(
    data: Float32Array,
    src_affine: Float64Array,
    dst_affine: Float64Array,
    spatial_size: Sequence[int],
) -> tuple[Float32Array, Float64Array, Resampling]:
    """``monai.transforms.spatial.functional.spatial_resample`` (bilinear, border, float64)."""
    rank = min(data.ndim - 1, len(src_affine) - 1, 3, len(spatial_size))
    source = to_affine_nd(rank, src_affine)
    target = to_affine_nd(rank, dst_affine)
    in_size = tuple(int(size) for size in data.shape[1 : rank + 1])
    size = tuple(int(value) for value in spatial_size[:rank])
    record = Resampling(source, in_size)
    xform = np.linalg.solve(source, target)
    same_size = np.allclose(size, in_size)
    unchanged = (np.allclose(source, target, atol=AFFINE_TOL) and same_size) or (
        np.allclose(xform, np.eye(len(xform)), atol=AFFINE_TOL) and same_size
    )
    if unchanged:
        return data.astype(np.float32), np.array(src_affine, dtype=np.float64), record
    values = _grid_sample(data, xform, in_size, size)
    affine = np.array(src_affine, dtype=np.float64)
    new_affine = _mm(affine, to_affine_nd(len(affine) - 1, xform))
    return values, new_affine, record


def scale_intensity_range(
    data: Float32Array,
    a_min: float,
    a_max: float,
    b_min: float,
    b_max: float,
) -> Float32Array:
    """``ScaleIntensityRange(..., clip=True)`` in float32."""
    result = (data - np.float32(a_min)) / np.float32(a_max - a_min)
    result = result * np.float32(b_max - b_min) + np.float32(b_min)
    return np.clip(result, np.float32(b_min), np.float32(b_max)).astype(np.float32, copy=False)


def softmax(data: Float32Array) -> Float32Array:
    """``Activations(softmax=True)`` over channels: ``exp(x - max) / sum`` in float32.

    Torch's vectorized exponential can differ from NumPy's by a float32 rounding unit.
    """
    shifted = np.exp(data - data.max(axis=0, keepdims=True))
    total = shifted[0].copy()
    for channel in shifted[1:]:
        total += channel
    return (shifted / total).astype(np.float32, copy=False)


def argmax(data: Float32Array) -> Float32Array:
    """``AsDiscrete(argmax=True)``: the first maximal channel, as float32 ``(1, ...)``."""
    return np.argmax(data, axis=0)[None].astype(np.float32)


def write_nifti(path: Path, data: HostArray, affine: Float64Array, dtype: npt.DTypeLike) -> None:
    """Cast and write a millimetre NIfTI with sform and qform code 1.

    The sform stores the affine within NIfTI precision. The qform cannot
    represent shear, which NiBabel removes when encoding that form.
    """
    array = np.asarray(data).astype(dtype, copy=False)
    matrix = to_affine_nd(3, affine)
    image = nib.Nifti1Image(array, matrix)
    image.header.set_xyzt_units("mm")
    image.set_sform(matrix, code=1)
    image.set_qform(matrix, code=1)
    nib.save(image, str(path))






def _normalize_transform(shape: Sequence[int]) -> Float64Array:
    """``monai.networks.utils.normalize_transform`` (``align_corners=False``)."""
    size = np.asarray(shape, dtype=np.float64)
    norm = size.copy()
    norm[norm <= 0.0] = 2.0
    norm = 2.0 / norm
    result = np.diag(np.concatenate((norm, np.ones(1, dtype=np.float64))))
    result[:-1, -1] = 1.0 / size - 1.0
    return result






def _theta(
    xform: Float64Array,
    in_size: Sequence[int],
    out_size: Sequence[int],
) -> Float64Array:
    """``AffineTransform`` normalization and reverse indexing of a voxel transform."""
    theta = _bmm(
        _bmm(_normalize_transform(in_size), xform),
        np.linalg.inv(_normalize_transform(out_size)),
    )
    rank = len(in_size)
    reverse = list(range(rank - 1, -1, -1))
    theta[:rank] = theta[reverse]
    theta[:, :rank] = theta[:, reverse]
    return theta[:rank]


def _grid_sample(
    data: Float32Array,
    xform: Float64Array,
    in_size: Sequence[int],
    out_size: Sequence[int],
) -> Float32Array:
    """``affine_grid`` then trilinear ``grid_sample`` (border, ``align_corners=False``).

    MONAI samples a float64 copy of the image and rounds the result to float32; here
    each output chunk is computed in float64 from the gathered float32 voxels, which
    gives the same values without a float64 copy of the whole image.
    The grid's four-term dot products use paired multiply-adds,
    ``fma(z, t2, x * t0) + fma(1, t3, y * t1)``. CPU provider rounding is bounded
    by the recorded preprocessing comparisons.
    """
    if len(in_size) != 3:
        raise ValueError("SwinUNETR bundle resampling is three-dimensional")
    theta = _theta(xform, in_size, out_size)
    depth, height, width = (int(size) for size in out_size)
    # Base grid: x follows the last data axis, z the first.
    x = np.broadcast_to(_linspace_from_neg_one(width), (width,))
    y = np.broadcast_to(_linspace_from_neg_one(height), (height,))
    z = np.broadcast_to(_linspace_from_neg_one(depth), (depth,))
    # Per output row: the (z, x) part of the dot product and the y part.
    zx = [_fma_array(z[:, None], theta[row, 2], x[None, :] * theta[row, 0]) for row in range(3)]
    ys = [y * theta[row, 1] + theta[row, 3] for row in range(3)]
    separable = all(
        theta[row, column] == 0 for row in range(3) for column in range(3) if row != column
    )
    result = np.empty((data.shape[0], depth, height, width), dtype=np.float32)
    step = max(1, _CHUNK_VOXELS // max(1, height * width * data.shape[0]))

    def fill(start: int) -> None:
        """Interpolate a contiguous output-depth chunk into the shared result."""
        stop = min(depth, start + step)
        if separable:
            gx = zx[0][0] + ys[0][0]
            gy = zx[1][0, 0] + ys[1]
            gz = zx[2][start:stop, 0] + ys[2][0]
            result[:, start:stop] = _sample_separable(data, gx, gy, gz)
        else:
            coordinates = [
                zx[row][start:stop, None, :] + ys[row][None, :, None] for row in range(3)
            ]
            result[:, start:stop] = _sample(data, *coordinates)

    # Chunks write disjoint output planes and NumPy releases the GIL in their
    # gathers and arithmetic, so threads change only the elapsed time.
    with ThreadPoolExecutor(max_workers=_RESAMPLE_THREADS) as pool:
        for chunk in [pool.submit(fill, start) for start in range(0, depth, step)]:
            chunk.result()
    return result


def _fma_array(left: Float64Array, right: float, addend: Float64Array) -> Float64Array:
    """Approximate fused ``left * right + addend`` using error-free transforms.

    The decomposition assumes arithmetic without overflow or harmful
    underflow; it is not a general replacement for hardware FMA.
    """
    left, addend = np.broadcast_arrays(np.asarray(left, np.float64), np.asarray(addend, np.float64))
    right_value = np.float64(right)
    product = left * right_value
    if right_value == 0 or not np.any(left):
        return product + addend
    error = _product_error(left, right_value, product)
    total = product + addend
    virtual = total - product
    sum_error = (product - (total - virtual)) + (addend - virtual)
    return total + (sum_error + error)


def _product_error(left: Float64Array, right: np.float64, product: Float64Array) -> Float64Array:
    """Dekker remainder ``left * right - product`` under its finite-range assumptions."""

    def split[T: (Float64Array, np.float64)](value: T) -> tuple[T, T]:
        """Split a floating value into high and low components for product error."""
        scaled = 134217729.0 * value
        high = scaled - (scaled - value)
        return cast(T, high), cast(T, value - high)

    left_high, left_low = split(left)
    right_high, right_low = split(right)
    return (
        ((left_high * right_high - product) + left_high * right_low) + left_low * right_high
    ) + left_low * right_low


def _source_index(coordinate: Float64Array, size: int) -> Float64Array:
    """``grid_sampler_compute_source_index`` for border padding, ``align_corners=False``."""
    unnormalized = ((coordinate + 1) * size - 1) / 2
    return np.minimum(np.float64(size - 1), np.maximum(unnormalized, 0.0))


def _corners(
    index: Float64Array,
) -> tuple[tuple[Int64Array, Int64Array], tuple[Float64Array, Float64Array]]:
    """Lower/upper voxel indices and their ATen weights ``(upper - i, i - lower)``."""
    low = np.floor(index)
    high = low + 1
    return (low.astype(np.int64), high.astype(np.int64)), (high - index, index - low)


# ATen's corner order (tnw, tne, tsw, tse, bnw, bne, bsw, bse) as (z, y, x) picks.
_CORNERS = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 0),
    (0, 1, 1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def _sample_separable(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """Trilinear corners for an axis-aligned grid, in ATen's weight and sum order."""
    depth, height, width = data.shape[1:]
    x_index, x_weight = _corners(_source_index(gx, width))
    y_index, y_weight = _corners(_source_index(gy, height))
    z_index, z_weight = _corners(_source_index(gz, depth))
    along_d, along_h, along_w = (
        (slice(None), None, None),
        (None, slice(None), None),
        (None, None, slice(None)),
    )
    result = np.zeros((data.shape[0], len(gz), len(gy), len(gx)), dtype=np.float64)
    # The eight corners use two z slabs and four (z, y) planes; gather each once.
    slabs = [data.take(np.minimum(z_index[cz], depth - 1), axis=1) for cz in (0, 1)]
    planes = [
        [slab.take(np.minimum(y_index[cy], height - 1), axis=2) for cy in (0, 1)] for slab in slabs
    ]
    for cz, cy, cx in _CORNERS:
        # (x * y) * z, left to right, as in grid_sampler_3d_cpu_impl.
        weight = (x_weight[cx][along_w] * y_weight[cy][along_h]) * z_weight[cz][along_d]
        zi, yi, xi = z_index[cz], y_index[cy], x_index[cx]
        values = planes[cz][cy].take(np.minimum(xi, width - 1), axis=3).astype(np.float64)
        inside = ((zi < depth)[along_d] & (yi < height)[along_h]) & (xi < width)[along_w]
        result += np.where(inside, values * weight, 0.0)
    return result


def _sample(
    data: Float32Array,
    gx: Float64Array,
    gy: Float64Array,
    gz: Float64Array,
) -> Float64Array:
    """Trilinear corners for a general grid, in ATen's weight and sum order."""
    depth, height, width = data.shape[1:]
    x_index, x_weight = _corners(_source_index(gx, width))
    y_index, y_weight = _corners(_source_index(gy, height))
    z_index, z_weight = _corners(_source_index(gz, depth))
    corners = [
        (z_index[cz], y_index[cy], x_index[cx], (x_weight[cx] * y_weight[cy]) * z_weight[cz])
        for cz, cy, cx in _CORNERS
    ]
    shape = np.broadcast_shapes(gx.shape, gy.shape, gz.shape)
    result = np.zeros((data.shape[0], *shape), dtype=np.float64)
    for zi, yi, xi, weight in corners:
        inside = (zi < depth) & (yi < height) & (xi < width)
        values = data[
            :,
            np.minimum(zi, depth - 1),
            np.minimum(yi, height - 1),
            np.minimum(xi, width - 1),
        ].astype(np.float64)
        result += np.where(inside, values * weight, 0.0)
    return result
