import numpy as np
from npnd import errors
from .one_hot import one_hot

def gather_nd(params, indices, batch_dims=0):
  params = np.asarray(params)
  indices = np.asarray(indices)
  if not np.issubdtype(params.dtype, np.number):
    return gather_nd_generic(params, indices, batch_dims=batch_dims)
  if batch_dims > 0:
    return gather_nd_batched(params, indices, batch_dims=batch_dims)
  if np.ndim(params) < 1:
    return errors.invalid_argument("params must be at least a vector")
  if np.ndim(indices) < 1:
    return errors.invalid_argument("indices must be at least a vector")
  # Calculate the number of dimensions in indices
  slice_dim = indices.shape[-1]
  if slice_dim > np.ndim(params):
    return errors.invalid_argument(
        "index innermost dimension length must be <= params rank; saw: ",
        slice_dim, " vs. ", np.ndim(params))
  outer_shape = indices.shape[:-1]
  inner_shape = params.shape[slice_dim:]
  result_shape = outer_shape + inner_shape
  # Calculate the number of elements that make up each slice of the
  # tensor.
  slice_size = int(np.prod(inner_shape)) # 1 if inner_shape is empty
  # Calculate the number of slices we'll be selecting.
  num_slices = int(np.prod(params.shape)) // slice_size
  # Reshape the incoming tensor into (num_slices, slice_size).
  params_mat = params.reshape((num_slices, slice_size))
  # Calculate the 1-dimensional indices necessary to select
  # the correct slices.
  strides_shape = params.shape[:slice_dim]
  strides = get_stride_sizes(strides_shape)
  indices_mat = flat_inner_dims(indices)
  indices_mat = (strides * indices_mat).sum(-1)
  # Select the slices we want, via onehot-matmul.
  hot = one_hot(indices_mat, num_slices)
  result = hot @ params_mat
  # Reshape the result back to the expected shape.
  return result.reshape(result_shape)

def gather_nd_generic(params, indices, batch_dims):
  # if params contains non-numbers, handle it specially, since it can't be multiplied
  # against onehot matrices.
  items = params.flat[:].tolist()
  ids = np.arange(np.prod(params.shape)).reshape(params.shape)
  out = gather_nd(ids, indices, batch_dims=batch_dims)
  final = np.asarray([items[i] for i in out.flat[:].tolist()]).reshape(out.shape)
  return final

def gather_nd_batched(params, indices, batch_dims):
  # TODO: Clean this up. I have a feeling it can be unified with the
  # logic in gather_nd.
  #
  # These shapes came from
  # https://www.tensorflow.org/api_docs/python/tf/gather_nd
  index_depth = indices.shape[-1]
  batch_shape = indices.shape[:batch_dims]
  assert params.shape[:batch_dims] == batch_shape
  outer_shape = indices.shape[batch_dims:-1]
  assert index_depth <= np.ndim(params)
  inner_shape = params.shape[batch_dims + index_depth:]
  result_shape = batch_shape + outer_shape + inner_shape
  # TODO: I'm only confident that the batch_dims==1 case works.
  assert batch_dims == 1, "batch_dims > 1 not yet implemented"
  batched_indices = add_batch_indices(indices)
  result = gather_nd(params, batched_indices, batch_dims=0)
  return result.reshape(result_shape)

def get_stride_sizes(shape):
  remain_flat_size = int(np.prod(shape)) # 1 if shape is empty
  dims_to_count = []
  for dim in shape:
    remain_flat_size //= dim
    dims_to_count += [remain_flat_size]
  return dims_to_count

def flat_inner_shape(shape, num_out_dims = 2):
  assert num_out_dims > 0
  out_dims = [0 for i in range(num_out_dims)]
  offset = len(shape) - num_out_dims
  for out_dim in reversed(range(num_out_dims)):
    in_dim = out_dim + offset
    out_dims[out_dim] = 1 if in_dim < 0 else shape[in_dim]
  for in_dim in range(offset):
    out_dims[0] *= shape[in_dim]
  return tuple(out_dims)

assert flat_inner_shape((4,4,4)) == (16, 4)
assert flat_inner_shape((4,4,4), 1) == (64,)
assert flat_inner_shape((4,4,4), 2) == (16, 4)
assert flat_inner_shape((4,4,4), 3) == (4, 4, 4)
assert flat_inner_shape((4,4,4), 4) == (1, 4, 4, 4)
assert flat_inner_shape((4,4,4), 5) == (1, 1, 4, 4, 4)

def flat_inner_dims(tensor, num_out_dims = 2):
  tensor = np.asarray(tensor)
  shape = flat_inner_shape(tensor.shape, num_out_dims)
  return tensor.reshape(shape)

def add_batch_indices(indices):
  indices = np.asarray(indices)
  ind = np.arange(indices.shape[0])
  #ind = ind.reshape(indices.shape)
  shape = (-1,) + tuple(1 for i in range(len(indices.shape) - 1))
  ind = ind.reshape(shape)
  ind = np.concatenate([ind, indices], -1)
  #ind = np.concatenate([indices[...,:-1], ind, indices[...,-1:]], -1)
  ind = np.expand_dims(ind, -2)
  return ind

