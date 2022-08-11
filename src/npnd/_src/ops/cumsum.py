import numpy as np
import math

from . import reshape as reshape_lib
from . import broadcast as broadcast_lib
from . import shape as shape_lib

def cumop(op, a, axis=None, dtype=None) -> np.ndarray:
  a = np.asanyarray(a, dtype=dtype)
  if axis is None:
    a = reshape_lib.flatten(a)
    axis = 0
  elif axis < 0:
    axis += shape_lib.ndim(a)
  if axis != shape_lib.ndim(a) - 1:
    a = a.swapaxes(axis, -1)
  shape = a.shape
  shape = list(shape)
  shape.append(shape[-1])
  a = reshape_lib.expand_dims(a, -1)
  a = broadcast_lib.broadcast_to(a, shape)
  a = shape_lib.triu(a)
  if op == np.prod:
    mask = shape_lib.tril(np.ones_like(a), -1)
    a = a + mask
  a = op(a, axis=-2)
  if axis != shape_lib.ndim(a) - 1:
    a = a.swapaxes(-1, axis)
  return a

def cumsum(a, axis=None, dtype=None) -> np.ndarray:
  return cumop(np.sum, a, axis=axis, dtype=dtype)

def cumprod(a, axis=None, dtype=None) -> np.ndarray:
  return cumop(np.prod, a, axis=axis, dtype=dtype)


def axis_index(shape, axis, index):
  idx = []
  for i in range(len(shape)):
    idx.append(slice(None, None))
  idx[axis] = index
  return tuple(idx)

def presum(a, axis=None, dtype=None) -> np.ndarray:
  a = np.asanyarray(a, dtype=dtype)
  if axis is None:
    a = reshape_lib.flatten(a)
    axis = 0
  elif axis < 0:
    axis += shape_lib.ndim(a)
  n = a.shape[axis]
  a = np.copy(a)
  steps = 0
  for i in range(int(math.ceil(math.log2(n)))):
    b = np.copy(a)
    steps1 = steps
    print(steps, range(int(math.ceil(math.log2(n)))), range(2**i, n))
    for j in range(2**i, n):
      dst = axis_index(a.shape, axis, j)
      src = axis_index(a.shape, axis, j - 2**i)
      print(i, j, dst, src)
      b[dst] = a[dst] + a[src]
      steps += 1
    print(steps - steps1)
    a = b
  print(steps)
  return a


def cumop(op, a, axis=None, dtype=None) -> np.ndarray:
  a = np.asanyarray(a, dtype=dtype)
  if axis is None:
    a = reshape_lib.flatten(a)
    axis = 0
  elif axis < 0:
    axis += shape_lib.ndim(a)
  if axis != shape_lib.ndim(a) - 1:
    a = a.swapaxes(axis, -1)
  shape = a.shape
  shape = list(shape)
  shape.append(shape[-1])
  a = reshape_lib.expand_dims(a, -1)
  a = broadcast_lib.broadcast_to(a, shape)
  a = shape_lib.triu(a)
  if op == np.prod:
    mask = shape_lib.tril(np.ones_like(a), -1)
    a = a + mask
  a = op(a, axis=-2)
  if axis != shape_lib.ndim(a) - 1:
    a = a.swapaxes(-1, axis)
  return a


