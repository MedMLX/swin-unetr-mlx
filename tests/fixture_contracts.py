"""Schemas for immutable upstream recordings and runner arguments."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from swin_unetr_mlx._records import ErrorBound, Upstream


class SavedFacts(TypedDict):
    path: str
    dtype: str
    shape: list[int]
    qform_code: int
    sform_code: int
    affine: list[list[float]]


class Case(TypedDict):
    case: str
    saved: SavedFacts


class Host(TypedDict):
    system: str
    machine: str


class Metadata(TypedDict):
    schema_version: int
    recording_host: Host
    upstream: Upstream
    bundle_revision: str
    cases: NotRequired[list[Case]]
    saved: NotRequired[SavedFacts]
    sample_stride: NotRequired[int]
    logits_tolerance: NotRequired[ErrorBound]


class RunnerMetadata(TypedDict):
    saved: SavedFacts


class NetworkMetadata(TypedDict):
    weights_seed: int
    sample_stride: int
    logits_tolerance: ErrorBound
