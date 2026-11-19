# Performance

The axis-aligned resampler gathers x columns once per z slab and reuses a
float64 buffer for the weighted corner products. This preserves the trilinear
corner order and applies during preprocessing and mask restoration.

Measured on an Apple M1 Max with 64 GiB of memory, macOS 27.2.1, MLX 0.32.4,
medmlx-core 0.3.0, and NumPy 2.5.4. Each run used the released SwinUNETR BTCV checkpoint
from `MedMLX/swin-unetr-mlx` at revision `0052abb4bfb1c56bd7d52d5353267393399b8743`,
with four 96×96×96 windows per batch, 50% overlap, and resident stitching.

| HCC-TACE-Seg CT | Input shape | Full process (s) | Inference (s) | Postprocessing and writing (s) | Peak MLX (GiB) | Peak process footprint (GiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HCC_001 | 512×512×36 | 14.790 → 14.325 | 11.825 → 11.806 | 2.432 → 1.994 | 6.46 → 6.46 | 21.68 → 21.36 |
| HCC_063 | 512×512×99 | 37.715 → 36.605 | 33.847 → 34.010 | 3.106 → 1.810 | 6.46 → 6.46 | 16.57 → 16.66 |

Values are medians of six alternating fresh-process pairs per CT, comparing
`548efc0` with the final implementation after one uncounted round. Full-process time
includes startup, CT and local checkpoint loading, inference, and the mask write.
MLX peak is the runner's allocator high-water mark during inference. Process
footprint is the macOS high-water mark over the full process.

The 96.9% paired intervals for the postprocessing and writing time were −21.0%
to −16.3% on HCC_001 and −42.3% to −40.7% on HCC_063. Full-process intervals
were −4.12% to −1.23% and −4.34% to −2.03%, respectively. The HCC_001 full-process
interval overlaps the 2% noise band. On both CTs, the stitched float32 logits
before softmax and the complete written NIfTI files matched the baseline bit
for bit.

## Restoring from cropped window logits

A CT whose resampled grid is shorter than the 96-voxel window on one axis is
padded for inference, and its stitched logits come back as a cropped view of the
padded array. `ndarray.take` makes a non-contiguous source C-contiguous before it
gathers, so each restoration chunk copied the whole 14-channel volume to read a
few planes. [NumPy 2.5.4 take implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/item_selection.c#L223-L237)
For a non-contiguous source the resampler now gathers each chunk's depth planes
from that chunk's own range of planes. Contiguous logits take the same path as
before.

Same M1 Max, MLX 0.32.4 and checkpoint as above. Values are medians of six
alternating fresh-process pairs per CT after one uncounted round, comparing the
column-gather implementation with and without the plane-range gather.

| HCC-TACE-Seg CT | Resampled grid | Full process (s) | Postprocessing and writing (s) | Peak MLX (GiB) | Peak process footprint (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| HCC_001 | 267×267×88, padded to 96 | 14.470 → 13.355 | 2.001 → 0.928 | 6.46 → 6.46 | 21.36 → 18.75 |
| HCC_063 | 294×294×124 | 37.000 → 37.075 | 1.843 → 1.818 | 6.46 → 6.46 | 16.64 → 16.64 |

On HCC_001 the 96.9% paired intervals were −8.8% to −4.7% for full-process
time, −54.6% to −52.8% for postprocessing and writing, and −12.3% to −12.2% for
peak process footprint. On HCC_063, whose logits are contiguous, the
full-process interval was −2.1% to +2.2% and the footprint interval −0.3% to
+0.1%. On each CT, all seven masks written with the plane-range gather,
including the warm-up, matched the baseline files byte for byte.

## Candidates not adopted

Two further changes were measured against `548efc0` in the same six-pair setup
and left out. Both kept the stitched logits and the written masks bit for bit.

Reusing the centered values inside instance normalization, which saves one
mean and one subtraction per normalization, raised peak MLX memory from 6.46 GiB
to 7.25 GiB on HCC_001 and to 7.70 GiB on HCC_063. Full-process time rose from
14.86 s to 15.50 s and from 38.05 s to 39.57 s.

Blending the previous window batch on the host while Metal runs the next one
gave 96.9% paired intervals for inference time of −3.97% to +0.65% on HCC_001
and −3.45% to +0.21% on HCC_063, which does not clear the 2% band. Peak MLX
memory went from 6.455 GiB to 6.469 GiB.

## MLX buffer cache

### Uncapped memory attribution

On the M1 Max setup above, the uncapped diagnostic ran each CT once through the
column-gather implementation, before the plane-range gather was added. It sampled MLX's current active and cached bytes, its
allocator high-water mark, and the macOS physical footprint about every 10 ms. Values
are GiB. The footprint column is a current reading at the named boundary or peak
sample. The MLX peak counter starts over at inference.

| CT | Stage | MLX active | MLX cache | MLX peak | Process footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| HCC_001 | After CT and weights load | 0.24 | 0.24 | 0.47 | 0.73 |
| HCC_001 | After inference | 0.42 | 16.49 | 6.46 | 18.28 |
| HCC_001 | At restoration peak | 0.42 | 16.49 | 6.46 | 20.96 |
| HCC_001 | After postprocessing | 0.42 | 16.49 | 6.46 | 17.86 |
| HCC_063 | After CT and weights load | 0.24 | 0.24 | 0.47 | 0.82 |
| HCC_063 | After inference | 0.42 | 14.51 | 6.46 | 16.50 |
| HCC_063 | At restoration peak | 0.42 | 14.51 | 6.46 | 16.65 |
| HCC_063 | After postprocessing | 0.42 | 14.51 | 6.46 | 16.60 |

At the process peaks, the cache counter was 79% and 87% of the footprint, respectively.
Subtracting the contemporaneous MLX active and cache counters leaves about 4.05 GiB on
HCC_001 and 1.73 GiB on HCC_063 for host arrays and other runtime memory. This difference
is an estimate because the allocator and macOS track memory at different boundaries.

HCC_001's resampled grid is 267×267×88 and is padded to 267×267×96. Its 25 windows leave
a final single-window batch after six batches of four. The cache grows from 14.33 GiB
after the last full batch to 16.49 GiB after that shorter batch. HCC_063 resamples to
294×294×124 and runs 72 windows in 18 full batches, reusing the same batch shapes.

The FP32 source CTs occupy 36 and 99 MiB. The preprocessed inputs occupy 24 and 41 MiB.
Resident stitching holds a 366 or 572 MiB logits array, plus a 26 or 41 MiB count
map during inference. Softmax reuses the logits allocation. The per-worker restoration
scratch includes FP32 slabs, columns, and rounded chunks and FP64 weighted terms and
sums. The restored float32 masks occupy 36 and 99 MiB.

Cropping HCC_001's padded logits leaves a strided 335 MiB view. NumPy 2.5.4's
`ndarray.take` normalizes its input to C-contiguous storage, so each restoration worker
made a full copy when gathering a z slab. HCC_063 already returns contiguous logits.
The carryover from the final single-window batch and the host contiguity copies explain
why the scan with fewer slices had the larger process peak. The plane-range gather
above removes the copies. [NumPy 2.5.4 take implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/_core/src/multiarray/item_selection.c#L223-L237)

### Paired cap sweep

On the same M1 Max, MLX 0.32.4 setup, the full runner used the column-gather
implementation, before the plane-range gather, for HCC_001 (512×512×36) and
HCC_063 (512×512×99).
The five candidates set the cache limit just around inference. Perfkit
ran six counted rounds in balanced order and one separate warmup per
setting and CT. Each scan ran in a fresh process. The table reports
medians and 96.9% paired sign intervals for full-run time relative to the
uncapped run. macOS `time -lp` measured wall time and lifetime peak physical
footprint. The runner reported inference time and peak MLX allocation.

| HCC-TACE-Seg CT | Cache limit | Full run (s) | Inference (s) | Peak MLX (GiB) | Peak process footprint (GiB) | Footprint change | Paired full-run interval |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HCC_001 | Uncapped | 14.490 | 11.936 | 6.455 | 21.359 | — | — |
| HCC_001 | 0 MiB | 19.170 | 16.660 | 6.455 | 6.621 | −69.0% | +29.50% to +34.43% |
| HCC_001 | 256 MiB | 17.835 | 15.326 | 6.455 | 6.779 | −68.3% | +19.85% to +26.19% |
| HCC_001 | 512 MiB | 17.365 | 14.836 | 6.455 | 7.039 | −67.0% | +17.47% to +26.05% |
| HCC_001 | 1 GiB | 17.735 | 15.220 | 6.455 | 7.290 | −65.9% | +20.96% to +23.85% |
| HCC_001 | 2 GiB | 17.145 | 14.623 | 6.455 | 8.863 | −58.5% | +16.59% to +21.70% |
| HCC_063 | Uncapped | 36.715 | 34.116 | 6.455 | 16.643 | — | — |
| HCC_063 | 0 MiB | 51.340 | 48.761 | 6.455 | 7.016 | −57.8% | +34.90% to +43.27% |
| HCC_063 | 256 MiB | 47.675 | 45.102 | 6.455 | 7.190 | −56.8% | +26.19% to +35.36% |
| HCC_063 | 512 MiB | 46.915 | 44.343 | 6.455 | 7.443 | −55.3% | +26.23% to +32.54% |
| HCC_063 | 1 GiB | 47.175 | 44.600 | 6.455 | 7.690 | −53.8% | +26.90% to +31.08% |
| HCC_063 | 2 GiB | 45.875 | 43.282 | 6.455 | 9.281 | −44.2% | +19.07% to +27.67% |

All 70 capped masks, including warmups, matched the corresponding uncapped
`mask.nii.gz` files byte for byte. The uncapped, 0 MiB, 256 MiB, 512 MiB,
and 1 GiB runs all reported exactly 6,931,518,304 bytes of peak MLX memory.
The 2 GiB cap reached a maximum 97.5 KiB higher on each CT.

No cap qualified. Median peak process footprint fell by 44.2% to 69.0%,
and every full-run interval lay above the allowed +3%. The fastest cap
on both CTs was 2 GiB. Its full-run intervals were +16.59% to +21.70% on
HCC_001 and +19.07% to +27.67% on HCC_063. The runtime retains its uncapped
cache policy.

### Remaining memory with a 2 GiB cap

One diagnostic run per CT applied a 2 GiB cap during inference and traced NumPy
allocations during restoration. Both written masks matched their uncapped diagnostic
outputs byte for byte. The largest sampled footprints occurred during inference, at
8.84 GiB on HCC_001 and 9.28 GiB on HCC_063. Each peak sample had 4.90 GiB of live MLX
allocations and 1.42 GiB in the MLX cache. The allocator high-water mark was 6.46 GiB on
both scans.

The restoration trace found eight simultaneous 335 MiB copies on HCC_001, totaling
2.62 GiB at the z-slab gather. HCC_063 made zero full-volume copies at that gather. The
largest observed sum of NumPy allocations created during restoration was 2.82 GiB on
HCC_001 and 0.29 GiB on HCC_063. The plane-range gather above removes those copies.
