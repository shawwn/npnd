import numpy as np
from functools import reduce

from npnd import core

def broadcast_to(array, shape, subok=False):
  if subok:
    array = np.asanyarray(array)
  else:
    array = np.asarray(array)
  return array * np.ones_like(array, shape=shape)

def broadcast_arrays(*args, subok=False):
  shape = broadcast_shapes(*(np.shape(a) for a in args))
  return tuple(broadcast_to(a, shape, subok=subok) for a in args)

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

# https://www.tensorflow.org/xla/operation_semantics#broadcastindim
# https://www.tensorflow.org/xla/broadcasting
def broadcast_in_dim(operand, shape, broadcast_dimensions):
  core._check_shapelike('broadcast_in_dim', 'shape', shape)
  core._check_shapelike('broadcast_in_dim', 'broadcast_dimensions', broadcast_dimensions)
  operand_ndim = np.ndim(operand)
  if operand_ndim != len(broadcast_dimensions):
    msg = ('broadcast_in_dim broadcast_dimensions must have length equal to '
           'operand ndim; got broadcast_dimensions {} for operand ndim {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim))
  if len(shape) < operand_ndim:
    msg = ('broadcast_in_dim target broadcast shape must have equal or higher rank '
           'to the operand shape; got operand ndim {} and target broadcast ndim {}.')
    raise TypeError(msg.format(operand_ndim, len(shape)))
  if not set(broadcast_dimensions).issubset(set(range(len(shape)))):
    msg = ('broadcast_in_dim broadcast_dimensions must be a subset of output '
           'dimensions, got {} for operand ndim {} and shape {}.')
    raise TypeError(msg.format(broadcast_dimensions, operand_ndim, shape))
  if not all(core.symbolic_equal_one_of_dim(operand.shape[i],
                                            [1, shape[broadcast_dimensions[i]]])
             for i in range(operand_ndim)):
    msg = (
        "broadcast_in_dim operand dimension sizes must either be 1, or be "
        "equal to their corresponding dimensions in the target broadcast "
        "shape; got operand of shape {}, target broadcast shape {}, "
        "broadcast_dimensions {} ")
    raise TypeError(msg.format(operand.shape, shape, broadcast_dimensions))
  if (len(broadcast_dimensions) != len(set(broadcast_dimensions)) or
      tuple(broadcast_dimensions) != tuple(sorted(broadcast_dimensions))):
    msg = ("broadcast_in_dim broadcast_dimensions must be strictly increasing; "
           "got broadcast_dimensions {}")
    raise TypeError(msg.format(broadcast_dimensions))
  in_reshape = np.ones(len(shape), dtype=np.int32)
  for i, bd in enumerate(broadcast_dimensions):
    in_reshape[bd] = operand.shape[i]
  return broadcast_to(np.reshape(operand, in_reshape), shape)
