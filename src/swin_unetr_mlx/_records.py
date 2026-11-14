"""Immutable source specification and public runner result schemas."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from medmlx_core.outputs import RunResult as _RunResult


class ErrorBound(TypedDict):
    atol: float
    rtol: float


class Qualification(TypedDict):
    seeded_logits: ErrorBound


class Upstream(TypedDict):
    package: str
    version: str
    revision: str


class Bundle(TypedDict):
    repo_id: str
    revision: str
    gated: bool
    weights: list[str]


class Parameter(TypedDict):
    shape: list[int]
    dtype: str


class Inference(TypedDict):
    labels: list[str]
    axcodes: str
    pixdim: list[float]
    intensity: list[float]
    roi_size: list[int]
    sw_batch_size: int
    overlap: float
    output_dtype: str
    output_postfix: str
    separate_folder: bool


class Specification(TypedDict):
    schema_version: int
    qualification: Qualification
    upstream: Upstream
    bundle: Bundle
    parameters: dict[str, Parameter]
    inference: Inference


class Runtime(TypedDict):
    """MLX device string and peak MLX memory in bytes (from ``mx.get_peak_memory``)."""

    device: str
    peak_bytes: int | None


class SlidingWindow(TypedDict):
    roi_size: list[int]
    sw_batch_size: int
    overlap: float


class Seconds(TypedDict):
    """Wall-clock seconds per runner stage, rounded to four decimals."""

    load_inputs_and_weights: float
    preprocess: float
    inference: float
    postprocess_and_write: float


class Details(TypedDict):
    """The ``details`` block of ``run``'s result.

    ``labels`` maps each label value (as a string) to its name and ``labels_present`` lists
    the values found in the written mask. ``input_shape`` is the input voxel shape (equal to
    the mask shape); ``network_input_shape`` is the spatial shape after reorientation and
    resampling to ``network_spacing_mm`` (mm). ``peak_mlx_bytes`` is peak MLX memory in
    bytes. ``requires_human_review`` is always true: outputs are research results.
    """

    weights_revision: NotRequired[str | None]
    streaming: bool
    bundle: str
    labels: dict[str, str]
    labels_present: list[int]
    input_shape: list[int]
    network_input_shape: list[int]
    network_spacing_mm: list[float]
    output_dtype: str
    sliding_window: SlidingWindow
    mlx_device: str
    peak_mlx_bytes: int | None
    seconds: Seconds
    requires_human_review: bool


# Result of ``swin_unetr_mlx.medmlx.run``: ``outputs["mask"]`` is the mask path.
type RunResult = _RunResult[Details]
