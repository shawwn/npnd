import numpy as np

from .one_hot import one_hot
from .gather_nd import get_stride_sizes

# like np.squeeze, but doesn't error if size != 1.
def maybe_squeeze(x, axis: int):
  x = np.asarray(x)
  if x.shape[axis] == 1:
    x = np.squeeze(x, axis)
  return x

def check(tensor, indices, updates):
  tensor = np.asarray(tensor)
  indices = np.asarray(indices)
  updates = np.asarray(updates)
  assert np.ndim(indices) >= 2
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  assert index_depth <= np.ndim(tensor)
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == batch_shape + inner_shape
  return tensor, indices, updates

def scatter_nd_slice_via_matmul(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  hot = one_hot(indices.T, tensor.shape[0])
  rhs = (updates.T @ hot).T
  mask = (np.ones_like(updates).T @ hot).T
  mask = (mask == 0)
  rhs = maybe_squeeze(rhs, -1)
  mask = maybe_squeeze(mask, -1)
  mask = 1 if reduction == 'add' else mask
  if reduction == 'mul':
    return tensor * (mask + rhs)
  return mask * tensor + rhs

def scatter_nd_slice_via_reduction(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  assert np.ndim(tensor) == 1
  # tensor.shape is [P]
  assert np.ndim(indices) == 2
  assert indices.shape[1] == 1
  # indices.shape is [N,1]
  assert np.ndim(updates) == 1
  # updates.shape is [N]
  hot = one_hot(indices.T, tensor.shape[0])
  # hot.shape is [1, N, P]
  updates = np.expand_dims(updates, axis=[0,-1])
  # updates.shape is now [1, N, 1]
  updates = np.broadcast_to(updates, hot.shape)
  # updates.shape is now [1, N, P]
  # make the last two axes unique, then smash them together.
  updates = updates * hot
  updates = updates.sum(axis=-2)
  # updates.shape is now [1, N]
  updates = updates.reshape(tensor.shape)
  # updates.shape is now [N]
  mask = (updates == 0)
  mask = 1 if reduction == 'add' else mask
  if reduction == 'mul':
    return tensor * (mask + updates)
  return mask * tensor + updates

scatter_nd_slice = scatter_nd_slice_via_matmul

def collapse(tensor, batch_shape):
  out = tensor.reshape((int(np.prod(batch_shape)),) + (tensor.shape[-1],))
  result = maybe_squeeze(out, -1)
  return result

def scatter_nd(tensor, indices, updates, reduction=None):
  tensor = np.asarray(tensor)
  indices = np.asarray(indices)
  updates = np.asarray(updates)
  assert np.ndim(indices) >= 2
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  assert index_depth <= np.ndim(tensor)
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == batch_shape + inner_shape
  if index_depth == 1:
    return scatter_nd_slice(tensor, indices, updates, reduction=reduction)
  stride_sizes = get_stride_sizes(outer_shape)
  tensor_shape = (np.prod(outer_shape),) + inner_shape
  ind = collapse(indices, batch_shape)
  lhs = (ind * stride_sizes).sum(-1)
  rhs = tensor.reshape(tensor_shape)
  upd = collapse(updates, batch_shape)
  lhs1 = np.expand_dims(lhs, -1)
  out = scatter_nd_slice(rhs, lhs1, upd, reduction=reduction)
  result = out.reshape(tensor.shape)
  return result
