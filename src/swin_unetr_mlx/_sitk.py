"""Typed protocol for the SimpleITK image IO calls used at the ITK boundary."""

from __future__ import annotations

from typing import Protocol, cast

import SimpleITK

from swin_unetr_mlx._arrays import HostArray


class Image(Protocol):
    def GetDirection(self) -> tuple[float, ...]:
        """Return the image direction cosine matrix in flattened order."""
        ...
    def GetSpacing(self) -> tuple[float, ...]:
        """Return physical spacing along the image axes."""
        ...
    def GetOrigin(self) -> tuple[float, ...]:
        """Return the physical origin of the image."""
        ...


class ImageFileReader(Image, Protocol):
    def SetFileName(self, name: str) -> None:
        """Select the source image."""
        ...
    def ReadImageInformation(self) -> None:
        """Read geometry and metadata without decoding pixels."""
        ...
    def GetSize(self) -> tuple[int, ...]:
        """Return the image size in ITK axis order."""
        ...


class SimpleITKModule(Protocol):
    def ImageFileReader(self) -> ImageFileReader:
        """Create a reader supporting header-only geometry inspection."""
        ...
    def ReadImage(self, fileName: str) -> Image:
        """Read an image from a file."""
        ...
    def GetArrayFromImage(self, image: Image) -> HostArray:
        """Copy image voxels into a NumPy array."""
        ...


sitk = cast(SimpleITKModule, SimpleITK)
