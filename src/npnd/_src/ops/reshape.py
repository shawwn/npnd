import numpy as np
import operator
import math

from typing import Sequence, Tuple

Shape = Sequence[int]

def listify(x):
  if isinstance(x, (list, tuple)):
    return list(x)
  elif x is None:
    return x
  else:
    return [x]

def prod(shape: Shape) -> int:
  # returns 1 if shape is empty
  # return int(np.prod(shape))
  return math.prod(shape)

def normalize_axis_index(axis: int, ndim: int, msg_prefix=None) -> int:
  # Check that index is valid, taking into account negative indices
  if axis < -ndim or axis >= ndim:
    raise ValueError((msg_prefix or "") + f" axis={axis!r} ndim={ndim!r}")
  # adjust negative indices
  if axis < 0:
    axis += ndim
  return axis

def normalize_axis_tuple(axis, ndim: int, argname=None, allow_duplicate=False) -> Tuple[int, ...]:
  # Optimization to speed-up the most common cases.
  if type(axis) not in (tuple, list):
    try:
      axis = [operator.index(axis)]
    except TypeError:
      pass
  # Going via an iterator directly is slower than via list comprehension.
  axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
  if not allow_duplicate and len(set(axis)) != len(axis):
    if argname:
      raise ValueError('repeated axis in `{}` argument'.format(argname))
    else:
      raise ValueError('repeated axis')
  return axis

def reshape_shape(input_shape: Shape, output_shape: Shape) -> Shape:
  spec = output_shape
  input_shape = list(input_shape)
  output_shape = list(output_shape)
  if spec.count(-1) > 1:
    raise ValueError("can only specify one unknown dimension")
  elif spec.count(-1) == 1:
    # infer unspecified value.
    auto_axis = spec.index(-1)
    auto_shape = list(spec)
    del auto_shape[auto_axis]
    assert prod(auto_shape) != 0
    output_shape[auto_axis] = prod(input_shape) // prod(auto_shape)
  # number of elements in the output shape should match the number of elements in the input shape.
  if prod(input_shape) != prod(output_shape):
    raise ValueError(f"Expected number of elements to be the same. {input_shape} {spec}")
  return output_shape

def reshape(tensor, output_shape: Shape) -> np.ndarray:
  tensor = np.asarray(tensor)
  tensor = tensor.copy()
  final_shape = reshape_shape(tensor.shape, output_shape)
  tensor.shape = final_shape
  return tensor

def expand_dims(a, axis: int) -> np.ndarray:
  a = np.asanyarray(a)
  if type(axis) not in (tuple, list):
    axis = (axis,)
  out_ndim = len(axis) + np.ndim(a)
  axis = normalize_axis_tuple(axis, out_ndim)
  shape = list(np.shape(a))
  shape = [1 if ax in axis else shape.pop(0) for ax in range(out_ndim)]
  return reshape(a, shape)

def flatten(a) -> np.ndarray:
  a = np.asanyarray(a)
  return reshape(a, [-1])
