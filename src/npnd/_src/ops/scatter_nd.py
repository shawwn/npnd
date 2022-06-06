import numpy as np

from .one_hot import one_hot
from .gather_nd import get_stride_sizes, flat_inner_dims

# like np.squeeze, but doesn't error if size != 1.
def maybe_squeeze(x, axis: int):
  x = np.asarray(x)
  if x.shape[axis] == 1:
    x = np.squeeze(x, axis)
  return x

def check(tensor, indices, updates):
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(float)
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
  print('')
  print('scatter_nd_slice_via_matmul  ', tensor.shape, indices.shape, updates.shape, reduction)
  hot = one_hot(indices.T, tensor.shape[0])
  print('scatter_nd_slice_via_matmul 2', tensor.shape, indices.shape, updates.shape, reduction)
  mask = (np.ones_like(updates).T @ hot).T
  mask = (mask == 0)
  mask = maybe_squeeze(mask, -1)
  # mask2 = (1.0 - hot).prod(-2).astype(bool)
  # mask2 = maybe_squeeze(mask2, -2)
  # mask2 = mask2.reshape(mask2.shape + tuple(1 for _ in range(len(tensor.shape) - 1)))
  # mask = mask2
  # if not np.array_equal(mask, mask2):
  #   breakpoint()
  # dupes = np.any(hot.prod(-2) > 0)
  # if dupes:
  #   breakpoint()
  # if reduction == 'mul':
  #   breakpoint()
  updates = (updates.T @ hot).T
  print('scatter_nd_slice_via_matmul 3', tensor.shape, indices.shape, updates.shape, reduction)
  updates = maybe_squeeze(updates, -1)
  print('scatter_nd_slice_via_matmul 4', tensor.shape, indices.shape, updates.shape, reduction)
  mask = 1 if reduction == 'add' else mask
  if reduction == 'mul':
    return tensor * (mask + updates)
  return mask * tensor + updates

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

def scatter_nd(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  print('')
  print('scatter_nd ', tensor.shape, indices.shape, updates.shape, reduction)
  index_depth = indices.shape[-1]
  outer_shape = tensor.shape[:index_depth]
  print('index_depth', index_depth)
  print('outer_shape', outer_shape)
  result_shape = tensor.shape
  if index_depth > 1:
    tensor = flat_inner_dims(tensor)
    updates = flat_inner_dims(updates)
    indices = flat_inner_dims(indices)
    stride_sizes = get_stride_sizes(outer_shape)
    indices = (indices * stride_sizes).sum(-1)
    indices = np.expand_dims(indices, axis=-1)
  result = scatter_nd_slice(tensor, indices, updates, reduction=reduction)
  result = result.reshape(result_shape)
  return result


def collapse(tensor, batch_shape):
  # out = tensor.reshape((int(np.prod(batch_shape)),) + (tensor.shape[-1],))
  out = tensor.reshape((int(np.prod(batch_shape)),) + (-1,))
  result = maybe_squeeze(out, -1)
  return result

def scatter_nd(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
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
  tensor_shape = (np.prod(outer_shape),) + ((np.prod(inner_shape),) if inner_shape else ())
  tensor_flat = tensor.reshape(tensor_shape)
  indices_flat = collapse(indices, batch_shape)
  indices_flat = (indices_flat * stride_sizes).sum(-1)
  updates_flat = collapse(updates, batch_shape)
  if reduction is None:
    # handle duplicate indices
    assert np.ndim(indices_flat) == 1
    duplicate_mask = indices_flat[:, None] == indices_flat[None, :]
    duplicate_mask = np.tril(duplicate_mask, -1).sum(0)
    duplicate_mask = duplicate_mask == 0
    # if np.any(duplicate_mask):
    #   breakpoint()
    while np.ndim(duplicate_mask) < np.ndim(updates_flat):
      duplicate_mask = np.expand_dims(duplicate_mask, -1)
    updates_flat = duplicate_mask * updates_flat
  indices_flat = np.expand_dims(indices_flat, -1)
  result = scatter_nd_slice(tensor_flat, indices_flat, updates_flat, reduction=reduction)
  result = result.reshape(tensor.shape)
  return result