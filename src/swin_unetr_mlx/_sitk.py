"""Typed protocol for the SimpleITK image IO calls used at the ITK boundary."""

from __future__ import annotations

from collections.abc import Sequence
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
    def SetDirection(self, direction: Sequence[float]) -> None:
        """Set the image direction cosine matrix."""
        ...
    def SetSpacing(self, spacing: Sequence[float]) -> None:
        """Set physical spacing along the image axes."""
        ...
    def SetOrigin(self, origin: Sequence[float]) -> None:
        """Set the physical origin of the image."""
        ...


class SimpleITKModule(Protocol):
    def ReadImage(self, fileName: str) -> Image:
        """Read an image from a file."""
        ...
    def GetArrayFromImage(self, image: Image) -> HostArray:
        """Copy image voxels into a NumPy array."""
        ...
    def GetImageFromArray(self, arr: HostArray) -> Image:
        """Create an image from a NumPy array."""
        ...
    def WriteImage(self, image: Image, fileName: str) -> None:
        """Write an image to a file."""
        ...
    def Version_VersionString(self) -> str:
        """Return the SimpleITK version string."""
        ...


sitk = cast(SimpleITKModule, SimpleITK)
