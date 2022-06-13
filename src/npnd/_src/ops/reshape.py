import numpy as np

def prod(shape):
  # returns 1 if shape is empty
  return int(np.prod(shape))

def reshape_shape(input_shape, spec):
  input_shape = list(input_shape)
  spec = list(spec)
  output_shape = list(spec)
  # spec must contain at most one -1 value.
  assert(spec.count(-1) <= 1)
  if spec.count(-1) == 1:
    # infer unspecified value.
    auto_axis = spec.index(-1)
    auto_shape = list(spec)
    del auto_shape[auto_axis]
    output_shape[auto_axis] = prod(input_shape) // prod(auto_shape)
  # number of elements in the output shape should match the number of elements in the input shape.
  if prod(input_shape) != prod(output_shape):
    raise ValueError(f"Expected number of elements to be the same. {input_shape} {spec}")
  return output_shape
