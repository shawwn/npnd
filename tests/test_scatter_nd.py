import unittest

import npnd
import numpy as np

# reference implementation of scatter_nd.

def scatternd_ref(original, indices, values, reduction=None):
  assert reduction in [None, 'add', 'mul'], "Unknown reduction type"
  original = np.asarray(original)
  indices = np.asarray(indices)
  values = np.asarray(values)
  output = np.copy(original)
  update_indices = indices.shape[:-1]
  for idx in np.ndindex(update_indices):
    if reduction is None:
      output[tuple(indices[idx])] = values[idx]
    elif reduction == 'add':
      output[tuple(indices[idx])] += values[idx]
    elif reduction == 'mul':
      output[tuple(indices[idx])] *= values[idx]
  return output
