import numpy as np

from . import reshape as reshape_lib
from . import broadcast as broadcast_lib
from . import gather_nd as gather_nd_lib

def diff(a, n=1, axis=-1, prepend=np._NoValue, append=np._NoValue):
  if n == 0:
    return a
  if n < 0:
    raise ValueError(
      "order must be non-negative but got " + repr(n))

  a = np.asanyarray(a)
  nd = a.ndim
  if nd == 0:
    raise ValueError("diff requires input that is at least one dimensional")
  axis = reshape_lib.normalize_axis_index(axis, nd)

  combined = []
  if prepend is not np._NoValue:
    prepend = np.asanyarray(prepend)
    if prepend.ndim == 0:
      shape = list(a.shape)
      shape[axis] = 1
      prepend = broadcast_lib.broadcast_to(prepend, tuple(shape))
    combined.append(prepend)

  combined.append(a)

  if append is not np._NoValue:
    append = np.asanyarray(append)
    if append.ndim == 0:
      shape = list(a.shape)
      shape[axis] = 1
      append = broadcast_lib.broadcast_to(append, tuple(shape))
    combined.append(append)

  if len(combined) > 1:
    a = np.concatenate(combined, axis)

  almost = np.arange(a.shape[axis] - 1)
  a1 = gather_nd_lib.take(a, 1 + almost, axis)
  a2 = gather_nd_lib.take(a, 0 + almost, axis)

  op = np.not_equal if a.dtype == np.bool_ else np.subtract
  for _ in range(n):
    a = op(a1, a2)

  return a