import numpy as np

def one_hot(indices, depth, axis=-1, dtype=np.int64):
  """Compute one hot from indices at a specific axis"""
  values = np.asarray(indices)
  rank = len(values.shape)
  depth_range = np.arange(depth)
  if axis < 0:
    axis += (rank + 1)
  ls = values.shape[0:axis]
  rs = values.shape[axis:rank]
  targets = np.reshape(depth_range, (1,) * len(ls) + depth_range.shape + (1,) * len(rs))
  values = np.reshape(values, ls + (1,) + rs)
  values = np.where(values < 0, values + depth, values) # wraparound negative indices
  # TODO: check whether indices are out of bounds?
  return np.asarray(targets == values, dtype=dtype)
