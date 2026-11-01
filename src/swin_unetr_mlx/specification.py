"""Immutable released-source graph specification."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from swin_unetr_mlx._records import Specification


def specification() -> Specification:
    """Read the packaged ``specification.json``: pinned upstream, tensor shapes, bundle settings."""
    return cast(Specification, json.loads(Path(__file__).with_suffix(".json").read_text()))


@dataclass(frozen=True, slots=True)
class BundleConfig:
    """The pinned bundle's inference settings, parsed at the resource boundary."""

    name: str
    labels: tuple[str, ...]
    axcodes: str
    pixdim: tuple[float, ...]
    intensity: tuple[float, ...]
    roi_size: tuple[int, ...]
    sw_batch_size: int
    overlap: float
    output_dtype: str
    output_postfix: str
    separate_folder: bool

    def __post_init__(self) -> None:
        """Validate the pinned bundle geometry and inference settings."""
        if (
            len(self.labels) != 14
            or len(self.pixdim) != 3
            or len(self.intensity) != 4
            or len(self.roi_size) != 3
            or self.sw_batch_size < 1
            or any(size < 1 for size in self.roi_size)
            or any(value <= 0 for value in self.pixdim)
            or not 0 <= self.overlap < 1
        ):
            raise ValueError("Invalid packaged SwinUNETR bundle settings")


def bundle_config() -> BundleConfig:
    """Parse the ``inference`` section of the packaged specification.

    ``pixdim`` is in mm, ``intensity`` is ``(a_min, a_max, b_min, b_max)`` with ``a_*`` in
    HU, ``roi_size`` in voxels at the resampled spacing, and ``overlap`` a fraction.
    """
    source = specification()
    if source["schema_version"] != 1:
        raise ValueError("Unsupported SwinUNETR specification schema")
    values = source["inference"]
    return BundleConfig(
        name=source["bundle"]["repo_id"].split("/")[-1],
        labels=tuple(values["labels"]),
        axcodes=values["axcodes"],
        pixdim=tuple(values["pixdim"]),
        intensity=tuple(values["intensity"]),
        roi_size=tuple(values["roi_size"]),
        sw_batch_size=values["sw_batch_size"],
        overlap=values["overlap"],
        output_dtype=values["output_dtype"],
        output_postfix=values["output_postfix"],
        separate_folder=values["separate_folder"],
    )
