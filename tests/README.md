The network and runner tests use the HCC-TACE-Seg HCC_063 contrast CT at
`~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`. The image-derived inputs and
MONAI reference outputs stay in `work/real-reference`, outside Git. The runner
fixtures use controlled test affines rather than the original CT geometry.

Generate the references with `scripts/record_network_reference.py` and
`scripts/record_runner_reference.py`, using MONAI 1.4.0, Torch 2.13, SimpleITK,
Einops, and the project test dependencies. Pass `--bundle` pointing to the
pinned `e3e994c2b1438f84f431162a6b319f2819500c65` BTCV bundle snapshot.
Then run `python -m pytest tests/test_network_reference.py tests/test_medmlx.py`
in the project environment.
