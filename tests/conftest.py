"""Isolate native crashes and fail native tests when the Metal probe fails."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from swin_unetr_mlx._types import MLX

import pytest


@pytest.fixture(scope="session")
def mx() -> MLX:
    """Probe and load the Apple Silicon Metal runtime for native tests."""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "from swin_unetr_mlx.runtime import import_mlx; import_mlx()"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(
            f"Native MLX probe timed out after {error.timeout} seconds; "
            f"stdout={error.stdout!r}; stderr={error.stderr!r}",
            pytrace=False,
        )
    if probe.returncode != 0:
        pytest.fail(
            f"Native MLX probe failed (exit {probe.returncode}); "
            f"stdout={probe.stdout!r}; stderr={probe.stderr!r}",
            pytrace=False,
        )
    from swin_unetr_mlx.runtime import import_mlx

    return import_mlx()
