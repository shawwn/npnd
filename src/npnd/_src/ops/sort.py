# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
import numpy as np

from . import shape as shape_lib
from . import gather_nd as gather_lib
from . import reshape as reshape_lib
from . import cumsum as cumsum_lib
from . import diff as diff_lib

def pow2_less_than(n):
  k = 1
  while k > 0 and k < n:
    k = k * 2
  return max(1, k // 2)

def swap_along_axis(a: np.ndarray, axis: int, lo: int, hi: int) -> np.ndarray:
  a = np.asarray(a)
  src = list(range(a.shape[axis]))
  src[lo] = hi
  src[hi] = lo
  out = gather_lib.take(a, src, axis=axis)
  return out

def bitonic_compare(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, hi: int, direction: bool) -> np.ndarray:
  idx_lo = gather_lib.take(idxs, [lo], axis=axis)
  idx_hi = gather_lib.take(idxs, [hi], axis=axis)
  a_lo = gather_lib.take_along_axis(a, idx_lo, axis=axis)
  a_hi = gather_lib.take_along_axis(a, idx_hi, axis=axis)
  mask = direction != np.all(a_lo >= a_hi)
  idxs_swapped = swap_along_axis(idxs, axis, lo, hi)
  out = np.where(mask, idxs, idxs_swapped)
  return out

def bitonic_merge(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool) -> np.ndarray:
  if n > 1:
    m = pow2_less_than(n)
    for i in range(n-m):
      idxs = bitonic_compare(axis, a, idxs, lo+i, lo+i+m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo, m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo+m, n-m, direction)
  return idxs

def bitonic_sort(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool) -> np.ndarray:
  if n > 1:
    m = n // 2
    idxs = bitonic_sort(axis, a, idxs, lo, m, not direction)
    idxs = bitonic_sort(axis, a, idxs, lo+m, n-m, direction)
    idxs = bitonic_merge(axis, a, idxs, lo, n, direction)
  return idxs

def argsort(a: np.ndarray, axis=-1, ascending=True) -> np.ndarray:
  a = np.asarray(a)
  n = a.shape[axis]
  idxs = shape_lib.ndshape(a.shape)[axis]
  out = bitonic_sort(axis, a, idxs, 0, n, direction=ascending)
  return out

def argmin(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  idxs = argsort(a, axis=axis)
  out = gather_lib.take(idxs, 0, axis=axis)
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def argmax(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  idxs = argsort(a, axis=axis)
  out = gather_lib.take(idxs, -1, axis=axis)
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def sort(a: np.ndarray, axis=-1, ascending=True) -> np.ndarray:
  idxs = argsort(a, axis=axis, ascending=ascending)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  return out

def amin(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  idxs = argmin(a, axis=axis, keepdims=True)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  if not keepdims:
    out = np.squeeze(out, axis)
  return out

def amax(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  idxs = argmax(a, axis=axis, keepdims=True)
  out = gather_lib.take_along_axis(a, idxs, axis=axis)
  if not keepdims:
    out = np.squeeze(out, axis)
  return out

def topk(X, k, axis, largest=False) -> np.ndarray:
  sorted_indices = argsort(X, axis=axis, ascending=not largest)
  sorted_values = sort(X, axis=axis, ascending=not largest)
  topk_sorted_indices = gather_lib.take(sorted_indices, np.arange(k), axis=axis)
  topk_sorted_values = gather_lib.take(sorted_values, np.arange(k), axis=axis)
  return topk_sorted_values, np.array(topk_sorted_indices, dtype=np.int64)

def unique(ar, return_index=False, return_inverse=False, return_counts=False, axis=None):
  ar = np.asanyarray(ar)
  if axis is None:
    ar = reshape_lib.flatten(ar)
    ret = _unique1d(ar, return_index, return_inverse, return_counts)
    return _unpack_tuple(ret)

  # axis was specified and not None
  try:
    ar = np.moveaxis(ar, axis, 0)
  except np.AxisError:
    # this removes the "axis1" or "axis2" prefix from the error message
    raise np.AxisError(axis, ar.ndim) from None

  # Must reshape to a contiguous 2D array for this to work...
  orig_shape, orig_dtype = ar.shape, ar.dtype
  ar = reshape_lib.reshape(ar, (orig_shape[0], np.prod(orig_shape[1:], dtype=np.intp)))
  ar = np.ascontiguousarray(ar)
  dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

  if False:
    # At this point, `ar` has shape `(n, m)`, and `dtype` is a structured
    # data type with `m` fields where each field has the data type of `ar`.
    # In the following, we create the array `consolidated`, which has
    # shape `(n,)` with data type `dtype`.
    try:
      if ar.shape[1] > 0:
        consolidated = ar.view(dtype)
      else:
        # If ar.shape[1] == 0, then dtype will be `np.dtype([])`, which is
        # a data type with itemsize 0, and the call `ar.view(dtype)` will
        # fail.  Instead, we'll use `np.empty` to explicitly create the
        # array with shape `(len(ar),)`.  Because `dtype` in this case has
        # itemsize 0, the total size of the result is still 0 bytes.
        consolidated = np.empty(len(ar), dtype=dtype)
    except TypeError as e:
      # There's no good way to do this for object arrays, etc...
      msg = 'The axis argument to unique is not supported for dtype {dt}'
      raise TypeError(msg.format(dt=ar.dtype)) from e
  else:
    consolidated = ar

  def reshape_uniq(uniq):
    n = len(uniq)
    uniq = uniq.view(orig_dtype)
    uniq = reshape_lib.reshape(uniq, (n, *orig_shape[1:]))
    uniq = np.moveaxis(uniq, 0, axis)
    return uniq

  output = _unique1d(consolidated, return_index,
                     return_inverse, return_counts)
  output = (reshape_uniq(output[0]),) + output[1:]
  return _unpack_tuple(output)

def _unique1d(ar, return_index=False, return_inverse=False, return_counts=False):
  """
  Find the unique elements of an array, ignoring shape.
  """
  # ar = reshape_lib.flatten(ar)

  optional_indices = return_index or return_inverse

  if optional_indices:
    perm = argsort(ar, axis=0)
    if perm.ndim > 1:
      assert perm.ndim == 2
      assert np.all(perm[..., 0:1] == perm)
      perm = gather_lib.take(perm, 0, axis=1)
    #aux = ar[perm]
    aux = gather_lib.take_along_axis(ar, perm, axis=0)
  else:
    ar = sort(ar)
    aux = ar
  mask = np.empty((aux.shape[0],), dtype=np.bool_)
  # mask = np.empty(aux.shape, dtype=np.bool_)
  mask[:1] = True
  if aux.shape[0] > 0 and aux.dtype.kind in "cfmM" and np.isnan(aux[-1]):
    raise NotImplementedError("can't do nan check")
    if aux.dtype.kind == "c":  # for complex all NaNs are considered equivalent
      aux_firstnan = np.searchsorted(np.isnan(aux), True, side='left')
    else:
      aux_firstnan = np.searchsorted(aux, aux[-1], side='left')
    if aux_firstnan > 0:
      mask[1:aux_firstnan] = (
              aux[1:aux_firstnan] != aux[:aux_firstnan - 1])
    mask[aux_firstnan] = True
    mask[aux_firstnan + 1:] = False
  else:
    result = (aux[1:] == aux[:-1])
    if result.ndim > 1:
      assert result.ndim == 2
      result = np.all(result, axis=-1)
      # result = result[..., 0]
    result = np.equal(result, False)
    mask[1:] = result

  #ret = (aux[mask],)
  ret = (gather_lib.take(aux, np.extract(mask, np.arange(aux.shape[0])), axis=0),)
  if return_index:
    #ret += (perm[mask],)
    #ret += (np.extract(mask, perm).reshape((-1,) + perm.shape[1:]),)
    ret += (np.extract(mask, perm),)
  if return_inverse:
    imask = cumsum_lib.cumsum(mask) - 1
    inv_idx = np.empty(mask.shape, dtype=np.intp)
    # inv_idx[perm] = imask
    np.put(inv_idx, perm, imask)
    ret += (inv_idx,)
  if return_counts:
    idx = np.concatenate(np.nonzero(mask) + ([mask.size],))
    ret += (diff_lib.diff(idx),)
  return ret

def _unpack_tuple(x):
  """ Unpacks one-element tuples for use as return values """
  if len(x) == 1:
    return x[0]
  else:
    return x