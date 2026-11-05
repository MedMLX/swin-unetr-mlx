"""Typed protocols for the NiBabel NIfTI, header and orientation calls used here."""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike
from typing import Protocol, cast

import nibabel
import numpy as np
from nibabel.filebasedimages import ImageFileError as ImageFileError

from swin_unetr_mlx._arrays import Float64Array, HostArray

type PathArg = str | PathLike[str]


class ArrayBuffer(Protocol):
    def __array__(
        self, dtype: np.dtype[np.generic] | None = None, copy: bool | None = None
    ) -> HostArray:
        """Convert the image buffer to a NumPy array with the requested dtype and copy policy."""
        ...


class NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel spacing stored in the NIfTI header."""
        ...
    def set_xyzt_units(self, xyz: str) -> None:
        """Set the NIfTI spatial unit name."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Read a named NIfTI header field."""
        ...


class NiftiImage(Protocol):
    affine: Float64Array
    dataobj: ArrayBuffer | HostArray
    header: NiftiHeader

    def __init__(self, dataobj: HostArray, affine: HostArray) -> None:
        """Construct an image from voxel data and an affine."""
        ...
    def set_sform(self, affine: Float64Array, code: int | None = None) -> None:
        """Set the sform affine and optional code."""
        ...
    def set_qform(self, affine: Float64Array, code: int | None = None) -> None:
        """Set the qform affine and optional code."""
        ...


class Orientations(Protocol):
    def axcodes2ornt(self, axcodes: str, labels: tuple[tuple[str, str], ...]) -> Float64Array:
        """Convert axis labels into an orientation transform."""
        ...
    def ornt_transform(self, start_ornt: Float64Array, end_ornt: Float64Array) -> Float64Array:
        """Compute the transform between source and target orientations."""
        ...
    def inv_ornt_aff(self, ornt: Float64Array, shape: Sequence[int]) -> Float64Array:
        """Return the inverse voxel affine for an orientation and shape."""
        ...
    def aff2axcodes(
        self, aff: Float64Array, labels: tuple[tuple[str, str], ...]
    ) -> tuple[str | None, ...]:
        """Determine axis labels from an affine and label pairs."""
        ...


class Nibabel(Protocol):
    Nifti1Image: type[NiftiImage]
    orientations: Orientations

    def load(self, filename: PathArg) -> NiftiImage:
        """Load a NIfTI image from a path."""
        ...
    def save(self, img: NiftiImage, filename: PathArg) -> None:
        """Save a NIfTI image to a path."""
        ...
    def io_orientation(self, affine: Float64Array) -> Float64Array:
        """Compute voxel-axis orientation from an affine."""
        ...


# Bound to the real module; callers validate shape and affine after loading.
nib = cast(Nibabel, nibabel)
