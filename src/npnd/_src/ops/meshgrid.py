from __future__ import annotations

import numpy as np
from typing import Literal

from . import broadcast

def meshgrid_shapes(*xi, indexing: Literal["xy", "ij"] = 'xy'):
  if indexing not in ['xy', 'ij']:
    raise ValueError(
      "Valid values for `indexing` are 'xy' and 'ij'.")

  ndim = len(xi)
  s0 = (1,) * ndim
  output = [np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1:]) for i, x in enumerate(xi)]

  if indexing == 'xy' and ndim > 1:
    # switch first and second axis
    output[0].shape = (1, -1) + s0[2:]
    output[1].shape = (-1, 1) + s0[2:]
  return output

def meshgrid(*xi, indexing: Literal["xy", "ij"] = 'xy'):
  shapes = meshgrid_shapes(*xi, indexing=indexing)
  output = broadcast.broadcast_arrays(*shapes, subok=True)
  return output
