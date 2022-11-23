from typing import *

import builtins as py
import numpy as np

from . import reshape as reshape_lib

def slice_on_axis(a: np.ndarray, axis: int, start: int = None, stop: int = None, step: int = None):
    axis = reshape_lib.normalize_axis_index(axis, len(np.shape(a)))
    t = [py.slice(None, None, None) for _ in range(len(np.shape(a)))]
    t[axis] = py.slice(start, stop, step)
    return a[t]

def slice_shape(a: np.ndarray, axis: Sequence[int], start: Sequence[int], stop: Sequence[int]):
    axis = reshape_lib.normalize_axis_tuple(axis, len(np.shape(a)), allow_duplicate=False)
    start = tuple(start)
    stop = tuple(stop)
    # assert isinstance(start, (tuple, list))
    # assert isinstance(stop, (tuple, list))
    assert len(start) == len(axis)
    assert len(stop) == len(axis)
    shape = list(np.shape(a))
    for i in range(len(axis)):
        dim = axis[i]
        n = shape[dim]
        lo = start[i]
        if lo < 0:
            lo += n
        hi = stop[i]
        if hi < 0:
            hi += n
        m = hi - lo
        if m < 0:
            m = 0
        if m > n:
            m = n
        shape[dim] = m
    return py.tuple(shape)

def slice(a: np.ndarray, axis: Sequence[int], start: Sequence[int], stop: Sequence[int]):
    a = np.asarray(a)
    axis = reshape_lib.normalize_axis_tuple(axis, len(a.shape), allow_duplicate=False)
    start = tuple(start)
    stop = tuple(stop)
    # assert isinstance(start, (tuple, list))
    # assert isinstance(stop, (tuple, list))
    assert len(start) == len(axis)
    assert len(stop) == len(axis)
    shape = slice_shape(a, axis, start, stop)
    out = a
    for dim, lo, hi in zip(axis, start, stop):
        out = slice_on_axis(out, dim, lo, hi)
    # assert shape == np.shape(out)
    return out
