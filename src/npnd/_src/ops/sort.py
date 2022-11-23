# https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
import numpy as np
import builtins as py

from . import shape as shape_lib
from . import gather_nd as gather_lib
from . import reshape as reshape_lib
from . import cumsum as cumsum_lib
from . import diff as diff_lib
from . import one_hot as one_hot_lib
from .trace import trace

from typing import Sequence, Tuple, Optional

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

def all(a: np.ndarray, axis: int, keepdims: bool):
  mask = np.asanyarray(a, dtype=np.bool_)
  return np.max(mask, axis=axis, keepdims=keepdims)

def slice_on_axis(a: np.ndarray, axis: int, start: int, stop: int):
  axis = reshape_lib.normalize_axis_index(axis, len(np.shape(a)))
  t = [py.slice(None, None, None) for _ in range(len(np.shape(a)))]
  t[axis] = py.slice(start, stop)
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


def bitonic_compare(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, hi: int, direction: bool) -> np.ndarray:
  with trace(f"bitonic_compare({axis=}, {a.shape=}, {idxs.shape=}, {lo=}, {hi=}, {direction=})"):
    if not direction:
      lo, hi = hi, lo
    if False:
      idx_lo = gather_lib.take(idxs, [lo], axis=axis)
      idx_hi = gather_lib.take(idxs, [hi], axis=axis)
      a_lo = gather_lib.take_along_axis(a, idx_lo, axis=axis)
      a_hi = gather_lib.take_along_axis(a, idx_hi, axis=axis)
    else:
      idx_lo = slice(idxs, [axis], [lo], [lo + 1])
      idx_hi = slice(idxs, [axis], [hi], [hi + 1])
      a_lo = gather_lib.gather_elements_ref(a, idx_lo, axis=axis)
      a_hi = gather_lib.gather_elements_ref(a, idx_hi, axis=axis)
    # cond = a_lo <= a_hi
    cond = a_lo < a_hi
    # mask = np.all(cond, axis=axis, keepdims=True)
    mask = all(cond, axis=axis, keepdims=True)
    assert (mask == cond).all()
    mask = cond
    idxs_swapped = swap_along_axis(idxs, axis, lo, hi)
    out = np.where(mask, idxs, idxs_swapped)
    trace(f"{idxs=}")
    trace(f"{idxs_swapped=}")
    trace(f"{cond.astype('i8')=}")
    trace(f"{mask.astype('i8')=}")
    trace(f"{out=}")
    return out

def bitonic_merge(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool) -> np.ndarray:
  with trace(f"bitonic_merge({axis=}, {a.shape=}, {idxs.shape=}, {lo=}, {n=}, {direction=})"):
    if n > 1:
      m = pow2_less_than(n)
      for i in range(n-m):
        idxs = bitonic_compare(axis, a, idxs, lo+i, lo+i+m, direction)
      idxs = bitonic_merge(axis, a, idxs, lo, m, direction)
      idxs = bitonic_merge(axis, a, idxs, lo+m, n-m, direction)
    return idxs

def bitonic_sort(axis: int, a: np.ndarray, idxs: np.ndarray, lo: int, n: int, direction: bool) -> np.ndarray:
  with trace(f"bitonic_sort({axis=}, {a.shape=}, {idxs.shape=}, {lo=}, {n=}, {direction=})"):
    if n > 1:
      m = n // 2
      idxs = bitonic_sort(axis, a, idxs, lo, m, not direction)
      idxs = bitonic_sort(axis, a, idxs, lo+m, n-m, direction)
      idxs = bitonic_merge(axis, a, idxs, lo, n, direction)
    return idxs

def bitonic_compare2(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], lo: int, hi: int, direction: bool, invert: bool) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  # with trace(f"bitonic_compare({vals=}, {idxs=}, {lo=}, {hi=}, {direction=})" and None):
  if True:
    if lo == hi:
      return vals, idxs, 0
    if not direction:
      lo, hi = hi, lo
    valA = vals[lo]
    valB = vals[hi]
    if idxs or invert:
      idxA = idxs[lo]
      idxB = idxs[hi]
    if invert:
      mask = np.less_equal(idxA, idxB)
    else:
      mask = np.less_equal(valA, valB)
    val_min = np.where(mask, valA, valB)
    val_max = np.where(mask, valB, valA)
    vals = list(vals)
    vals[lo] = val_min
    vals[hi] = val_max
    vals = tuple(vals)
    if idxs:
      idx_min = np.where(mask, idxA, idxB)
      idx_max = np.where(mask, idxB, idxA)
      idxs = list(idxs)
      idxs[lo] = idx_min
      idxs[hi] = idx_max
      idxs = tuple(idxs)
    return vals, idxs, 1

def bitonic_slice2(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], lo: int, hi: int, count: int, direction: bool, invert: bool) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  total = 0
  if not direction:
    lo, hi = hi, lo
    direction = True
  trace(f"bitonic_slice({lo=}, {hi=}, {count=})")
  for i in range(count):
    # print(" ", f"{lo+i=}, {hi+i=}, {direction=}")
    vals, idxs, k = bitonic_compare2(vals, idxs, lo+i, hi+i, direction, invert)
    total += k
  return vals, idxs, total

def bitonic_slice3(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], lo: int, hi: int, count: int, direction: bool, invert: bool) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  if count <= 1:
    return bitonic_slice2(vals, idxs, lo, hi, count, direction, invert)
  if not direction:
    lo, hi = hi, lo
  trace(f"bitonic_slice({lo=}, {hi=}, {count=})")
  valA = shape_lib.stack(vals[lo:lo+count], axis=0)
  valB = shape_lib.stack(vals[hi:hi+count], axis=0)
  mask = np.less_equal(valA, valB)
  val_min = np.where(mask, valA, valB)
  val_max = np.where(mask, valB, valA)
  vals = list(vals)
  vals[lo:lo+count] = shape_lib.unstack(val_min, axis=0)
  vals[hi:hi+count] = shape_lib.unstack(val_max, axis=0)
  vals = tuple(vals)
  if idxs:
    idxA = shape_lib.stack(idxs[lo:lo+count], axis=0)
    idxB = shape_lib.stack(idxs[hi:hi+count], axis=0)
    idx_min = np.where(mask, idxA, idxB)
    idx_max = np.where(mask, idxB, idxA)
    idxs = list(idxs)
    idxs[lo:lo+count] = shape_lib.unstack(idx_min, axis=0)
    idxs[hi:hi+count] = shape_lib.unstack(idx_max, axis=0)
    idxs = tuple(idxs)
  return vals, idxs, 1

def bitonic_merge2(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], lo: int, n: int, direction: bool, invert: bool) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  # if True:
  with trace(f"bitonic_merge({lo=}, {n=}, {direction=})" if n > 1 else None):
    total = 0
    if n > 1:
      m = pow2_less_than(n)
      # hi = lo+m
      # if not direction:
      #   lo, hi = hi, lo
      # vals, idxs, k1 = bitonic_slice2(vals, idxs, lo, lo+m, n-m, direction, invert)
      # vals, idxs, k1 = bitonic_slice3(vals, idxs, lo, lo+m, n-m, direction, invert)
      # if direction:
      #   vals, idxs, k1 = bitonic_slice3(vals, idxs, lo, lo+m, n-m, True)
      # else:
      #   vals, idxs, k1 = bitonic_slice3(vals, idxs, lo+m, lo, n-m, True)
      # # vals, idxs, k1 = bitonic_slice2(vals, idxs, lo, hi, n-m, True)
      # vals, idxs, k1 = bitonic_slice3(vals, idxs, lo, hi, n-m, True)

      # vals, idxs, k1 = bitonic_slice3(vals, idxs, lo, lo+m, n-m, direction, invert)
      vals, idxs, k1 = bitonic_slice2(vals, idxs, lo, lo+m, n-m, direction, invert)
      vals, idxs, k3 = bitonic_merge2(vals, idxs, lo+m, n-m, direction, invert)
      vals, idxs, k2 = bitonic_merge2(vals, idxs, lo, m, direction, invert)
      total += k1 + k2 + k3
    return vals, idxs, total

def bitonic_sort2(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], lo: int, n: int, direction: bool, invert: bool) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  # if True:
  with trace(f"bitonic_sort({lo=}, {n=}, {direction=})" if n > 1 else None):
    total = 0
    if n > 1:
      m = n // 2
      vals, idxs, k2 = bitonic_sort2(vals, idxs, lo+m, n-m, direction, invert)
      vals, idxs, k1 = bitonic_sort2(vals, idxs, lo, m, not direction, invert)
      vals, idxs, k3 = bitonic_merge2(vals, idxs, lo, n, direction, invert)
      total += k1 + k2 + k3
    return vals, idxs, total

KEEPDIMS = False

def nsorted(a: np.ndarray, axis=0, ascending=True, indices=True, invert=False) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
  vals = shape_lib.unstack(a, axis=axis, keepdims=KEEPDIMS)
  idxs = ()
  if isinstance(indices, np.ndarray):
    idxs = indices
  elif indices:
    idxs = shape_lib.ndshape(np.shape(a))[axis]
  if len(idxs):
    idxs = shape_lib.unstack(idxs, axis=axis, keepdims=KEEPDIMS)
  n = np.shape(a)[axis]
  vals, idxs, total = bitonic_sort2(vals, idxs, 0, n, direction=ascending, invert=invert)
  return vals, idxs, total

def sorted(a: np.ndarray, axis=0, ascending=True, indices=True, invert=False) -> Tuple[np.ndarray, Optional[np.ndarray], int]:
  vals, idxs, total = nsorted(a, axis=axis, ascending=ascending, indices=indices, invert=invert)
  vals = shape_lib.stack(vals, axis=axis)
  if idxs:
    idxs = shape_lib.stack(idxs, axis=axis)
  else:
    idxs = None
  return vals, idxs, total

def unsorted(vals: np.ndarray, idxs: np.ndarray, axis=0, ascending=True, indices=True) -> np.ndarray:
  # n = np.shape(idxs)[axis]
  # vals = shape_lib.unstack(vals, axis=axis, keepdims=False)
  # idxs = shape_lib.unstack(idxs, axis=axis, keepdims=False)
  # vals, idxs, total = bitonic_sort2(idxs, vals, 0, n, direction=not ascending, invert=True)
  # a = shape_lib.stack(vals, axis=axis)
  # return a
  a, _, total = sorted(vals, indices=idxs, axis=axis, ascending=ascending, invert=True)
  return a, total

def nunsorted(vals: np.ndarray, idxs: np.ndarray, axis=0, ascending=True, indices=True) -> np.ndarray:
  a, _, total = nsorted(vals, indices=idxs, axis=axis, ascending=ascending, invert=True)
  return a, total

# def topk(X, k, axis, largest=False):
#   with trace(f"topk({np.shape(X)=}, {k=}, {axis=}, {largest=}"):
#     sorted_indices = argsort(X, axis=axis, ascending=not largest)
#     # sorted_values = sort(X, axis=axis, ascending=not largest)
#     # sorted_values = gather_lib.take_along_axis(X, sorted_indices, axis=axis)
#     sorted_values = gather_lib.gather_elements(X, sorted_indices, axis=axis)
#     topk_sorted_indices = gather_lib.take(sorted_indices, np.arange(k), axis=axis)
#     topk_sorted_values = gather_lib.take(sorted_values, np.arange(k), axis=axis)
#     return topk_sorted_values, np.array(topk_sorted_indices, dtype=np.int64)

def topk(X, k, axis, largest=False):
  sorted_vals, sorted_idxs, total = nsorted(X, axis=axis, ascending=not largest)
  vals = shape_lib.stack(sorted_vals[0:k], axis=axis)
  idxs = shape_lib.stack(sorted_idxs[0:k], axis=axis).astype(np.int64)
  return vals, idxs

def argsort(a: np.ndarray, axis=-1, ascending=True) -> np.ndarray:
  # a = np.asarray(a)
  # n = a.shape[axis]
  # idxs = shape_lib.ndshape(a.shape)[axis]
  # breakpoint()
  # out = bitonic_sort(axis, a, idxs, 0, n, direction=ascending)
  vals, idxs, total = sorted(a, axis=axis, ascending=ascending)
  return idxs

def sort(a: np.ndarray, axis=-1, ascending=True) -> np.ndarray:
  # idxs = argsort(a, axis=axis, ascending=ascending)
  # out = gather_lib.take_along_axis(a, idxs, axis=axis)
  # return out
  vals, idxs, total = sorted(a, axis=axis, ascending=ascending, indices=False)
  return vals

def argmin(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  # idxs = argsort(a, axis=axis)
  # out = gather_lib.take(idxs, 0, axis=axis)
  # if keepdims:
  #   out = np.expand_dims(out, axis)
  vals, idxs, total = nsorted(a, axis=axis, ascending=True)
  out = idxs[0]
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def argmax(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  # idxs = argsort(a, axis=axis)
  # out = gather_lib.take(idxs, -1, axis=axis)
  # if keepdims:
  #   out = np.expand_dims(out, axis)
  # return out
  vals, idxs, total = nsorted(a, axis=axis, ascending=False)
  out = idxs[0]
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def amin(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  # idxs = argmin(a, axis=axis, keepdims=True)
  # out = gather_lib.take_along_axis(a, idxs, axis=axis)
  # if not keepdims:
  #   out = np.squeeze(out, axis)
  # return out
  vals, idxs, total = nsorted(a, axis=axis, ascending=True, indices=False)
  out = vals[0]
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

def amax(a: np.ndarray, axis=-1, keepdims=False) -> np.ndarray:
  # idxs = argmax(a, axis=axis, keepdims=True)
  # out = gather_lib.take_along_axis(a, idxs, axis=axis)
  # if not keepdims:
  #   out = np.squeeze(out, axis)
  # return out
  vals, idxs, total = nsorted(a, axis=axis, ascending=False, indices=False)
  out = vals[0]
  if keepdims:
    out = np.expand_dims(out, axis)
  return out

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

def ndstack(shape):
  return shape_lib.stack(shape_lib.ndshape(shape), axis=-1)

def gather_nd_via_sort(v: np.ndarray, ndindices: np.ndarray):
  info = gather_lib.gather_nd_info(v, ndindices)
  vflat = v.reshape(info.params_shape)
  ndflat = shape_lib.flat_nd_indices(ndindices, info.strides_shape)
  # add missing indices
  missing = np.setdiff1d(np.arange(info.num_slices), ndflat)
  ndfull = np.concatenate([ndflat, missing])
  _, indices, _ = sorted(ndfull)
  gathered, total = nunsorted(vflat, indices)
  sliced = gathered[0:len(ndflat)]
  stacked = shape_lib.stack(sliced)
  # TODO: handle repeated indices
  return np.reshape(stacked, info.result_shape), total

