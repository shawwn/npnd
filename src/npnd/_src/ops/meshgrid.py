from __future__ import annotations

import numpy as np
import math
from typing import Literal, Sequence, List, Tuple

from . import broadcast as broadcast_lib

Shape = Sequence[int]

def as_range(x, dtype=None) -> np.ndarray:
  if isinstance(x, slice):
    start = 0 if x.start is None else x.start
    step = 1 if x.step is None else x.step
    return np.arange(start, x.stop, step, dtype=dtype)
  elif isinstance(x, tuple):
    return np.arange(*x, dtype=dtype)
  elif not isinstance(x, np.ndarray):
    return np.arange(x, dtype=dtype)
  else:
    return np.asanyarray(x, dtype=dtype)

def meshgrid_axis(shape: Shape, axis: int, dtype=None):
  out = as_range(shape[axis], dtype=dtype)
  dims = [1] * len(shape)
  dims[axis] = -1
  out = np.reshape(out, dims)
  return out

def iota(shape: Shape, axis: int, dtype=None):
  #out = ogrid[tuple(map(slice, shape))][axis]
  # return ndshape(shape, dtype=dtype)[axis]
  out = meshgrid_axis(shape, axis, dtype=dtype)
  out = np.broadcast_to(out, shape)
  return out

def meshgrid_axes(shape: Shape, dtype=None, indexing: Literal["xy", "ij"] = 'ij'):
  if indexing not in ['xy', 'ij']:
    raise ValueError(
      "Valid values for `indexing` are 'xy' and 'ij'.")
  output = [meshgrid_axis(shape, axis, dtype=dtype) for axis in range(len(shape))]
  if indexing == 'xy' and len(output) > 1:
    # switch first and second axis
    output[0] = output[0].swapaxes(0, 1)
    output[1] = output[1].swapaxes(0, 1)
  return output

def meshgrid_shapes(*xi, indexing: Literal["xy", "ij"] = 'xy') -> List[np.ndarray]:
  return meshgrid_axes(tuple(xi), indexing=indexing)
  # if indexing not in ['xy', 'ij']:
  #   raise ValueError(
  #     "Valid values for `indexing` are 'xy' and 'ij'.")
  #
  # ndim = len(xi)
  # s0 = (1,) * ndim
  # # output = [(np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]
  # output = [as_range(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]
  #
  # if indexing == 'xy' and ndim > 1:
  #   # switch first and second axis
  #   output[0].shape = (1, -1) + s0[2:]
  #   output[1].shape = (-1, 1) + s0[2:]
  # return output

def meshgrid(*xi, indexing: Literal["xy", "ij"] = 'xy', broadcast=True) -> List[np.ndarray]:
  axes = meshgrid_axes(xi, indexing=indexing)
  if not broadcast:
    return axes
  output = broadcast_lib.broadcast_arrays(*axes, subok=True)
  return output

def ndshape(shape: Shape, dtype=None, broadcast=True) -> List[np.ndarray]:
  #return meshgrid(*[np.arange(i, dtype=dtype) for i in shape], indexing='ij')
  return meshgrid(*[as_range(dim, dtype=dtype) for dim in shape], indexing='ij', broadcast=broadcast)

def ndcoords(shape: Shape, dtype=None) -> np.ndarray:
  return np.stack(ndshape(shape, dtype=dtype), -1)

def ndindex(shape: Shape, dtype=None):
  coords = ndcoords(shape, dtype=dtype)
  coords = np.reshape(coords, (-1, len(shape)))
  return [tuple(x) for x in coords]

class ogrid:
  def __class_getitem__(cls, item):
    if not isinstance(item, (list, tuple)):
      item = tuple([item])
    #return meshgrid_shapes(*item, indexing='ij')
    return meshgrid_axes(item)

class mgrid:
  def __class_getitem__(cls, item):
    out = ogrid[item]
    out = np.broadcast_arrays(*out)
    return np.stack(out, 0)

class _Indexable(object):
  """Helper object for building indexes for indexed update functions.

  .. deprecated:: 0.2.22
     Prefer the use of :attr:`jax.numpy.ndarray.at`. If an explicit index
     is needed, use :func:`jax.numpy.index_exp`.

  This is a singleton object that overrides the :code:`__getitem__` method
  to return the index it is passed.

  >>> jax.ops.index[1:2, 3, None, ..., ::2]
  (slice(1, 2, None), 3, None, Ellipsis, slice(None, None, 2))
  """
  __slots__ = ()
  def __getitem__(self, index):
    return index

index = _Indexable()

def one_hot(indices, depth: int, axis: int = -1, dtype=np.int64, wraparound_negative=True):
  indices = np.asanyarray(indices, dtype=dtype)
  if axis < 0:
    rank = len(indices.shape)
    axis += (rank + 1)
  shape = list(indices.shape)
  shape.insert(axis, depth)
  if wraparound_negative:
    # wraparound negative indices
    indices = np.where(indices < 0, indices + depth, indices)
  hot = iota(shape, axis, dtype=dtype)
  hot = hot == np.expand_dims(indices, axis)
  hot = hot.astype(dtype)
  return hot