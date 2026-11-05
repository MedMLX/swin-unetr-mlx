"""Host array dtypes at model, checkpoint and image boundaries."""

from __future__ import annotations

import numpy as np

type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]
type Float32Array = np.ndarray[tuple[int, ...], np.dtype[np.float32]]
type Float64Array = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
type Int64Array = np.ndarray[tuple[int, ...], np.dtype[np.int64]]

type NumericScalar = (
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
)
type NumericArray = np.ndarray[tuple[int, ...], np.dtype[NumericScalar]]

type Int32Array = np.ndarray[tuple[int, ...], np.dtype[np.int32]]
type UInt8Array = np.ndarray[tuple[int, ...], np.dtype[np.uint8]]
