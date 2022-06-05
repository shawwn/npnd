import numpy as np

#def batch_scatter_update(ref, indices, updates, use_locking=True, name=None):
def batch_scatter_update_indices(indices, axis=-1):
  """Generalization of `tf.compat.v1.scatter_update` to axis different than 0.

  Analogous to `batch_gather`. This assumes that `ref`, `indices` and `updates`
  have a series of leading dimensions that are the same for all of them, and the
  updates are performed on the last dimension of indices. In other words, the
  dimensions should be the following:

  `num_prefix_dims = indices.ndims - 1`
  `batch_dim = num_prefix_dims + 1`
  `updates.shape = indices.shape + var.shape[batch_dim:]`

  where

  `updates.shape[:num_prefix_dims]`
  `== indices.shape[:num_prefix_dims]`
  `== var.shape[:num_prefix_dims]`

  And the operation performed can be expressed as:

  `var[i_1, ..., i_n, indices[i_1, ..., i_n, j]] = updates[i_1, ..., i_n, j]`

  When indices is a 1D tensor, this operation is equivalent to
  `tf.compat.v1.scatter_update`.

  To avoid this operation there would be 2 alternatives:
  1) Reshaping the variable by merging the first `ndims` dimensions. However,
     this is not possible because `tf.reshape` returns a Tensor, which we
     cannot use `tf.compat.v1.scatter_update` on.
  2) Looping over the first `ndims` of the variable and using
     `tf.compat.v1.scatter_update` on the subtensors that result of slicing the
     first
     dimension. This is a valid option for `ndims = 1`, but less efficient than
     this implementation.

  See also `tf.compat.v1.scatter_update` and `tf.compat.v1.scatter_nd_update`.

  Args:
    ref: `Variable` to scatter onto.
    indices: Tensor containing indices as described above.
    updates: Tensor of updates to apply to `ref`.
    use_locking: Boolean indicating whether to lock the writing operation.
    name: Optional scope name string.

  Returns:
    Ref to `variable` after it has been modified.

  Raises:
    ValueError: If the initial `ndims` of `ref`, `indices`, and `updates` are
        not the same.
  """
  indices = np.asarray(indices)
  indices_shape = indices.shape
  indices_dimensions = np.ndim(indices)
  if axis < 0:
    axis += indices_dimensions

  if indices_dimensions is None:
    raise ValueError("batch_gather does not allow indices with unknown "
                     "shape.")

  nd_indices = np.expand_dims(indices, axis=-1)
  nd_indices_list = []

  # Scatter ND requires indices to have an additional dimension, in which the
  # coordinates of the updated things are specified. For this to be adapted to
  # the scatter_update with several leading dimensions, we simply make use of
  # a tf.range for all the leading dimensions followed by concat of all the
  # coordinates we created with the original indices.

  # For example if indices.shape = [2, 3, 4], we should generate the following
  # indices for tf.compat.v1.scatter_nd_update:
  # nd_indices[:, :, 0] = [[0, 0, 0], [1, 1, 1]]
  # nd_indices[:, :, 1] = [[0, 1, 2], [0, 1, 2]]
  # nd_indices[:, :, 2] = indices
  for dimension in range(indices_dimensions - 1):
    # In this loop we generate the following for the example (one for each
    # iteration).
    # nd_indices[:, :, 0] = [[0, 0, 0], [1, 1, 1]]
    # nd_indices[:, :, 1] = [[0, 1, 2], [0, 1, 2]]
    # This is done at every iteration with a tf.range over the size of the
    # i-th dimension and using broadcasting over the desired shape.
    dimension_size = indices_shape[dimension]
    shape_to_broadcast = [1] * (indices_dimensions + 1)
    shape_to_broadcast[dimension] = dimension_size
    dimension_range = np.arange(dimension_size, dtype=nd_indices.dtype).reshape(shape_to_broadcast)
    nd_indices_list.append(dimension_range * np.ones_like(nd_indices))
  # Add the original indices at the end, as described above, and concat.
  nd_indices_list.append(nd_indices)
  final_indices = np.concatenate(nd_indices_list, axis=-1)
  return final_indices
  # return scatter_nd_update(
  #     ref, final_indices, updates, use_locking=use_locking)

def batch_scatter_shape(indices_shape, dtype=None):
  indices_shape = list(indices_shape)
  indices_dimensions = len(indices_shape)

  nd_indices_shape = indices_shape
  nd_indices_list = []

  # Scatter ND requires indices to have an additional dimension, in which the
  # coordinates of the updated things are specified. For this to be adapted to
  # the scatter_update with several leading dimensions, we simply make use of
  # a tf.range for all the leading dimensions followed by concat of all the
  # coordinates we created with the original indices.

  # For example if indices.shape = [2, 3, 4], we should generate the following
  # indices for tf.compat.v1.scatter_nd_update:
  # nd_indices[:, :, 0] = [[0, 0, 0], [1, 1, 1]]
  # nd_indices[:, :, 1] = [[0, 1, 2], [0, 1, 2]]
  # nd_indices[:, :, 2] = indices
  for dimension in range(indices_dimensions):
    # In this loop we generate the following for the example (one for each
    # iteration).
    # nd_indices[:, :, 0] = [[0, 0, 0], [1, 1, 1]]
    # nd_indices[:, :, 1] = [[0, 1, 2], [0, 1, 2]]
    # This is done at every iteration with a tf.range over the size of the
    # i-th dimension and using broadcasting over the desired shape.
    dimension_size = indices_shape[dimension]
    shape_to_broadcast = [1] * indices_dimensions
    shape_to_broadcast[dimension] = dimension_size
    dimension_range = np.arange(dimension_size, dtype=dtype).reshape(shape_to_broadcast)
    nd_indices_list.append(dimension_range * np.ones(nd_indices_shape, dtype=dtype))
  return tuple(nd_indices_list)

def batch_scatter_shape(indices_shape, dtype=None):
  return np.meshgrid(*[np.arange(i, dtype=dtype) for i in indices_shape], indexing='ij')

def batch_gather_nd_indices(indices, batch_dims=1):
  indices = np.asarray(indices)
  indices_nd = list(batch_scatter_shape(indices.shape, dtype=indices.dtype))[0:batch_dims]
  indices_nd.append(indices)
  return np.stack(indices_nd, axis=-1)

def batch_scatter_indices(indices, axis=0):
  indices = np.asarray(indices)
  indices_nd = list(batch_scatter_shape(indices.shape, dtype=indices.dtype))
  indices_nd[axis] = indices
  return np.stack(indices_nd, axis=-1)

def batch_gather(params, indices, axis=0):
  indices_nd = batch_scatter_indices(indices, axis=axis)
  from ..gather_nd import gather_nd
  return gather_nd(params, indices_nd)