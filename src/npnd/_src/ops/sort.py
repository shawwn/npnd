# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
import numpy as np

from .tensorflow.state_ops import batch_scatter_shape as ndshape
from .tensorflow.state_ops import batch_scatter_indices as ndindices
from . import gather_nd as gather_lib

def pow2_less_than(n):
  k = 1
  while k > 0 and k < n:
    k = k * 2
  return max(1, k // 2)

def swap_along_axis(a: np.ndarray, axis: int, lo: int, hi: int):
  a = np.asarray(a)
  src = list(range(a.shape[axis]))
  src[lo] = hi
  src[hi] = lo
  out = gather_lib.take(a, src, axis=axis)
  return out

def bitonic_compare(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, hi: int, direction: bool):
  idx_lo = gather_lib.take(idxs, [lo], axis=axis)
  idx_hi = gather_lib.take(idxs, [hi], axis=axis)
  a_lo = gather_lib.take_along_axis(a, idx_lo, axis=axis)
  a_hi = gather_lib.take_along_axis(a, idx_hi, axis=axis)
  mask = direction == (a_lo < a_hi)
  idxs_swapped = swap_along_axis(idxs, axis, lo, hi)
  out = np.where(mask, idxs, idxs_swapped)
  return out

def bitonic_merge(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool):
  if n > 1:
    m = pow2_less_than(n)
    for i in range(n-m):
      idxs = bitonic_compare(axis, a, idxs, lo+i, lo+i+m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo, m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo+m, n-m, direction)
  return idxs

def bitonic_sort(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool):
  if n > 1:
    m = n // 2
    idxs = bitonic_sort(axis, a, idxs, lo, m, not direction)
    idxs = bitonic_sort(axis, a, idxs, lo+m, n-m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo, n, direction)
  return idxs

def argsort(a: np.ndarray, axis=-1, ascending=True):
  a = np.asarray(a)
  n = a.shape[axis]
  idxs = ndshape(a.shape)[axis]
  out = bitonic_sort(axis, a, idxs, 0, n, direction=ascending)
  return out

def argmin(a: np.ndarray, axis=-1, keepdims=False):
  idxs = argsort(a, axis=axis)
  out = gather_lib.take(idxs, 0, axis=axis)
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def argmax(a: np.ndarray, axis=-1, keepdims=False):
  idxs = argsort(a, axis=axis)
  out = gather_lib.take(idxs, -1, axis=axis)
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def sort(a: np.ndarray, axis=-1, ascending=True):
  idxs = argsort(a, axis=axis, ascending=ascending)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  return out

def amin(a: np.ndarray, axis=-1, keepdims=False):
  idxs = argmin(a, axis=axis, keepdims=True)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  if not keepdims:
    out = np.squeeze(out, axis)
  return out

def amax(a: np.ndarray, axis=-1, keepdims=False):
  idxs = argmax(a, axis=axis, keepdims=True)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  if not keepdims:
    out = np.squeeze(out, axis)
  return out

def topk(X, k, axis, largest=False):  # type: ignore
    sorted_indices = argsort(X, axis=axis, ascending=not largest)
    sorted_values = sort(X, axis=axis, ascending=not largest)
    topk_sorted_indices = gather_lib.take(sorted_indices, np.arange(k), axis=axis)
    topk_sorted_values = gather_lib.take(sorted_values, np.arange(k), axis=axis)
    return topk_sorted_values, np.array(topk_sorted_indices, dtype=np.int64)
