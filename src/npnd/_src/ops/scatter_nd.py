import numpy as np

from . import one_hot as one_hot_lib
from . import shape as shape_lib
from . import matmul as matmul_lib

import math

# reference implementation of scatter_nd.
def scatter_nd_ref(original, indices, values, reduction=None):
  original = np.asarray(original).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  values = np.asarray(values).astype(float)
  output = np.copy(original)
  update_indices = indices.shape[:-1]
  for idx in reversed(shape_lib.ndindex(update_indices)):
    if reduction is None:
      output[tuple(indices[idx])] = values[idx]
    elif reduction == 'add':
      output[tuple(indices[idx])] += values[idx]
    elif reduction in ['mul', 'multiply']:
      output[tuple(indices[idx])] *= values[idx]
    else:
      raise ValueError(f"Unknown reduction {reduction!r}")
  return output

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

from . import matmul as matmul_lib

def uniqify_updates(tensor, hot, indices, updates):
  if True:
    dupes = hot.cumsum(-2) >= 2
    hot = np.where(dupes, 0, hot)
    blanks = hot.sum(-1)
    #blanks = maybe_squeeze(blanks, 0)
    while 1 in blanks.shape:
      blanks = np.squeeze(blanks, blanks.shape.index(1))
    while np.ndim(blanks) < np.ndim(updates):
      blanks = np.expand_dims(blanks, -1)
    # while np.ndim(blanks) > np.ndim(updates):
    #   blanks = np.squeeze(blanks, blanks.shape.index(1))
    updates1 = blanks * updates
    if np.shape(updates1) != np.shape(updates):
      breakpoint()
    # hot = ((hot.cumsum(0) >= 2) == 0) * hot
    return tensor, hot, updates1
    # hot = ((hot.cumsum(0) >= 2) == 0) * hot
    # dedup = (hot.cumsum(-2) >= 2) == False

  # counts = hot.sum(-2, keepdims=True)
  shape = shape_lib.ndshape(indices.shape)[0]
  # if np.any(counts > 1):
  #   breakpoint()
  if True:
    dedup = np.zeros_like(indices, shape=np.shape(shape))
    for i in range(indices.shape[0]):
      dup = indices[i] == np.where(shape < i, indices, -1)
      print(dup.shape)
      dedup += dup
  elif True:
    dedup = np.sum([indices[i] == np.where(shape < i, indices, -1)  for i in range(indices.shape[0])], 0)
  elif False:
    #counts = hot.sum(axis + 1, keepdims=True)
    counts = hot.sum(-2, keepdims=True)
    dedup1 = counts == 1
    dedup2 = counts > 1
    dedup3 = shape_lib.expand(dedup2, hot.shape)
    dedup4 = shape_lib.expand(dedup2, hot.shape, reverse=True)
    if np.any(counts > 1):
      breakpoint()
    # dedup2 = np.flip(dedup2, axis=axis)
    dedup = dedup1 + dedup3
    hot = dedup * hot
  else:
    dedup = indices == indices.T
    print('\n',dedup.shape)
    dedup = np.tril(dedup, -1)
    dedup = dedup.sum(0)
    #print(dedup.shape)
  dedup = dedup == 0
  dedup = maybe_squeeze(dedup, -1)
  while np.ndim(dedup) < np.ndim(updates):
    dedup = np.expand_dims(dedup, -1)
  updates = dedup * updates
  return tensor, hot, updates

def blend(mask, tensor, updates, reduction):
  mask = maybe_squeeze(mask, -1)
  updates = maybe_squeeze(updates, -1)
  mask = 1 if reduction == 'add' else mask
  if reduction in ['mul', 'multiply']:
    return tensor * (mask + updates)
  return mask * tensor + updates

def scatter_nd_slice_via_matmul(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  hot = one_hot_lib.one_hot(indices.T, tensor.shape[0])
  if reduction is None:
    tensor, hot, updates = uniqify_updates(tensor, hot, indices, updates)

  mask = (np.ones_like(updates).T @ hot).T
  mask = (mask == 0)

  # updates = (updates.T @ hot).T
  updates = (updates.T.astype(np.float16) @ hot.astype(np.float16)).T.astype(np.float32)
  return blend(mask, tensor, updates, reduction)

def scatter_nd_slice_via_reduction0(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  hot = one_hot_lib.one_hot(indices.T, tensor.shape[0])
  if reduction is None:
    tensor, hot, updates = uniqify_updates(tensor, hot, indices, updates)
  hot = shape_lib.flat_inner_dims(hot)
  hot = np.expand_dims(hot, -2)

  counts = hot.sum(0, keepdims=True)
  blanks = counts == 0
  mask = blanks.T
  while np.ndim(mask) > len(tensor.shape):
    mask = np.squeeze(mask, -1)
  while np.ndim(mask) < len(tensor.shape):
    mask = np.expand_dims(mask, -1)
  mask = np.broadcast_to(mask, tensor.shape)

  updates = shape_lib.flat_outer_dims(updates)
  updates = np.expand_dims(updates, -1)
  updates = hot * updates
  if reduction == 'mul':
    updates = updates + (1 - hot) * (1 - blanks)
    updates = updates.prod(0)
  else:
    updates = updates.sum(0)
  updates = updates.T
  updates = updates.reshape(tensor.shape)
  return blend(mask, tensor, updates, reduction)

def scatter_nd_slice_via_reduction(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  hot = one_hot_lib.one_hot(indices.T, tensor.shape[0])
  if reduction is None:
    tensor, hot, updates = uniqify_updates(tensor, hot, indices, updates)
  mask = hot.sum(-2)
  mask = mask.T
  mask = (mask == 0)
  while np.ndim(mask) > len(tensor.shape):
    mask = np.squeeze(mask, -1)
  while len(mask.shape) < len(tensor.shape):
    mask = np.expand_dims(mask, -1)
  mask = np.broadcast_to(mask, tensor.shape)
  hot = shape_lib.flat_inner_dims(hot)
  hot = np.expand_dims(hot, -2)
  updates = shape_lib.flat_outer_dims(updates)
  updates = np.expand_dims(updates, -1)
  updates = hot * updates
  if reduction in ['mul', 'multiply']:
    counts = hot.sum(0, keepdims=True)
    blanks = (counts == 0).astype(hot.dtype)
    updates = updates + (1 - hot) * (1 - blanks)
    updates = updates.prod(0)
    if False:
      # updates = tf.where(tf.math.is_inf(updates), tf.cast(1, updates.dtype), updates)
      inf = tf.cast(float('inf'), updates.dtype)
      ninf = tf.cast(-float('inf'), updates.dtype)
      updates = tf.where(updates == inf, tf.cast(1, updates.dtype), updates)
      updates = tf.where(updates == ninf, tf.cast(1, updates.dtype), updates)
      updates = tf.where(updates != updates, tf.cast(1, updates.dtype), updates)
  else:
    updates = updates.sum(0)
  updates = updates.T
  updates = updates.reshape(tensor.shape)
  return blend(mask, tensor, updates, reduction=reduction)


def scatter_nd_slice_via_reduction_flat(tensor, indices, updates, reduction=None):
  #tensor, indices, updates = check(tensor, indices, updates)
  tensor = np.asarray(tensor).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(float)
  assert np.ndim(tensor) == 1
  # tensor.shape is [P]
  assert np.ndim(indices) == 2
  assert indices.shape[1] == 1
  # indices.shape is [N,1]
  assert np.ndim(updates) == 1
  # updates.shape is [N]
  hot = one_hot_lib.one_hot(indices.T, tensor.shape[-1])
  # hot.shape is [1, N, P]
  updates = np.expand_dims(updates, axis=[0,-1])
  # updates.shape is now [1, N, 1]
  updates = np.broadcast_to(updates, hot.shape)
  # updates.shape is now [1, N, P]
  # make the last two axes unique, then smash them together.
  updates = updates * hot
  updates = updates.sum(axis=-2)
  # updates.shape is now [1, P]
  updates = updates.reshape(tensor.shape)
  # updates.shape is now [P]
  if False:
    mask = (updates == 0)
  else:
    mask = hot.sum(-2) == 0
  mask = 1 if reduction == 'add' else mask
  if reduction == 'mul':
    return tensor * (mask + updates)
  return mask * tensor + updates

# scatter_nd_slice = scatter_nd_slice_via_matmul
scatter_nd_slice = scatter_nd_slice_via_reduction

def scatter_nd(tensor, indices, updates, reduction=None):
  tensor, indices, updates = check(tensor, indices, updates)
  assert np.ndim(indices) >= 2
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:-1]
  assert index_depth <= np.ndim(tensor)
  outer_shape = tensor.shape[:index_depth]
  inner_shape = tensor.shape[index_depth:]
  assert updates.shape == batch_shape + inner_shape
  tensor_shape = (math.prod(outer_shape),) + ((math.prod(inner_shape),) if inner_shape else ())
  updates_shape = (math.prod(batch_shape),) + ((math.prod(inner_shape),) if inner_shape else ())
  indices_shape = (math.prod(batch_shape),) + (index_depth,)
  tensor_flat = tensor.reshape(tensor_shape)
  updates_flat = updates.reshape(updates_shape)
  indices_flat = indices.reshape(indices_shape)
  stride_sizes = shape_lib.get_stride_sizes(outer_shape)
  indices_flat = (indices_flat * stride_sizes).sum(-1, keepdims=True)
  result = scatter_nd_slice(tensor_flat, indices_flat, updates_flat, reduction=reduction)
  result = result.reshape(tensor.shape)
  result = result.astype(tensor.dtype)
  return result

