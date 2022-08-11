import numpy as np

from . import gather_nd
from . import scatter_nd
from . import shape as shape_lib

def scatter(params, indices, updates, axis=0, reduction=None):
  indices_nd = shape_lib.ndindices(indices, axis=axis)
  result = scatter_nd.scatter_nd(params, indices_nd, updates, reduction=reduction)
  return result.reshape(params.shape)

def scatter_ref(tensor, indices, updates, axis=0, reduction=None):
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(float)
  out = np.copy(tensor)
  for src in reversed(shape_lib.ndindex(indices.shape)):
    dst = list(src)
    dst[axis] = indices[src]
    dst = tuple(dst)
    if reduction is None:
      out[dst] = updates[src]
    elif reduction == 'add':
      out[dst] += updates[src]
    elif reduction in ['mul', 'multiply']:
      out[dst] *= updates[src]
    elif reduction == 'min':
      out[dst] = np.minimum(out[dst], updates[src])
    elif reduction == 'max':
      out[dst] = np.maximum(out[dst], updates[src])
    else:
      raise ValueError(f"unknown reduction {reduction!r}")
  return out
