import numpy as np

from .tensorflow import state_ops
from . import gather_nd
from . import scatter_nd

def _nd_indices(indices):
  indices = np.asarray(indices)
  return state_ops.batch_scatter_update_indices(np.expand_dims(indices, -1))[..., 1:].squeeze(-2)[..., ::-1]

def _nd_indices(indices, axis=0):
  indices = np.asarray(indices)
  return state_ops.batch_scatter_indices(indices, axis=axis)

def _nd_flat_indices(indices, axis=0):
  indices = np.asarray(indices)
  ind = _nd_indices(indices, axis=axis)
  strides = gather_nd.get_stride_sizes(ind.shape[:-1])
  ind = strides * gather_nd.flat_inner_dims(ind)
  ind = ind.sum(-1)
  ind = ind[..., None]
  return ind

def scatter(params, indices, updates, axis=0, reduction=None):
  params = np.asarray(params)
  indices = np.asarray(indices)
  updates = np.asarray(updates)
  # ind = _nd_indices(indices)
  #indices_flat = (gather_nd.get_stride_sizes(ind.shape[:-1]) * gather_nd.flat_inner_dims(ind)).sum(-1)[..., None]
  indices_flat = _nd_flat_indices(indices, axis=axis)
  result = scatter_nd.scatter_nd(params.flat, indices_flat, updates.flat, reduction=reduction)
  return result.reshape(params.shape)
