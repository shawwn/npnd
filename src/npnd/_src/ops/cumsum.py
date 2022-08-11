import numpy as np

from . import reshape as reshape_lib
from . import broadcast as broadcast_lib
from . import shape as shape_lib

def cumop(op, a, axis=None, dtype=None):
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

def cumsum(a, axis=None, dtype=None):
  return cumop(np.sum, a, axis=axis, dtype=dtype)

def cumprod(a, axis=None, dtype=None):
  return cumop(np.prod, a, axis=axis, dtype=dtype)


