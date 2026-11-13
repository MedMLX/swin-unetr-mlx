"""HCC-TACE source inputs used by the SwinUNETR parity tests."""

from pathlib import Path

import numpy as np

from swin_unetr_mlx._arrays import Float32Array
from swin_unetr_mlx._nifti import nib


def network_input() -> Float32Array:
    """Crop HCC_063 contrast CT and scale the bundle's HU interval to [0, 1]."""
    path = Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    ct = np.asarray(nib.load(path).dataobj, dtype=np.float32)[100:196, 180:276, 2:98]
    if ct.shape != (96, 96, 96):
        raise ValueError("HCC-TACE CT does not cover the reference crop")
    return np.clip((ct + np.float32(175)) / np.float32(425), 0, 1)[None, None]
