import numpy as np
from functools import reduce

def broadcast_to(array, shape):
  return array * np.ones_like(array, shape=shape)

def broadcast_arrays(*args):
  shape = broadcast_shapes(*(np.shape(a) for a in args))
  return tuple(broadcast_to(a, shape) for a in args)

def broadcast_shapes(*args):
  return reduce(broadcast_shape, args, ())

def broadcast_shape(shape1, shape2):
  # To compute the result broadcasted shape, we compare operand shapes
  # element-wise: starting with the trailing dimensions, and working the
  # way backward. Two dimensions are compatible when
  #   1. they are equal, or
  #   2. one of them is 1
  # The result shape has the maximum among the two inputs at every
  # dimension index.
  result_shape = list(max(shape1, shape2, key=len))
  try:
    for i1, i2, iR in zip(*(reversed(range(len(shape))) for shape in (shape1, shape2, result_shape))):
      result_shape[iR] = broadcast_dim(shape1[i1], shape2[i2])
  except ValueError:
    raise ValueError(f"operands could not be broadcast together with shapes {shape1!r} {shape2!r}")
  return tuple(result_shape)

def broadcast_dim(v1, v2):
  if v1 == -1 or v2 == -1:
    # One or both dimensions is unknown. Follow TensorFlow behavior:
    #   - If either dimension is greater than 1, we assume that the program is
    #     correct, and the other dimension will be broadcast to match it.
    #   - If either dimension is 1, the other dimension is the output.
    if v1 > 1:
      return v1
    elif v2 > 1:
      return v2
    elif v1 == 1:
      return v2
    elif v2 == 1:
      return v1
    else:
      return -1
  else:
    if v1 == v2 or v2 == 1:
      return v1
    elif v1 == 1:
      return v2
    else:
      # This dimension of the two operand types is incompatible.
      raise ValueError(f"Can't broadcast dimension of size {v1} to a dimension of size {v2}")
