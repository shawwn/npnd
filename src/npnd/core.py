from typing import Union, Any, Sequence, Dict, Tuple
import builtins
import operator

import numpy as np

from ._src.config import config

def unzip2(pairs):
  lst1, lst2 = [], []
  for x1, x2 in pairs:
    lst1.append(x1)
    lst2.append(x2)
  return lst1, lst2

unsafe_map = builtins.map
def map(f, *xs):
  return list(unsafe_map(f, *xs))

unsafe_zip = builtins.zip
def zip(*args):
  fst, *rest = args = map(list, args)
  n = len(fst)
  for arg in rest:
    assert len(arg) == n
  return list(unsafe_zip(*args))


def _ensure_index(x: Any) -> Union[int, Tuple[int, ...]]:
  """Ensure x is either an index or a tuple of indices."""
  try:
    return operator.index(x)
  except TypeError:
    return tuple(map(operator.index, x))

def _ensure_index_tuple(x: Any) -> Tuple[int, ...]:
  """Convert x to a tuple of indices."""
  try:
    return (operator.index(x),)
  except TypeError:
    return tuple(map(operator.index, x))


### Operations on shapes and dimension sizes.

# Shapes are tuples of dimension sizes, which are normally integers. We allow
# modules to extend the set of dimension sizes to contain other types, e.g.,
# symbolic dimensions in jax2tf.shape_poly.DimVar and masking.Poly.
DimSize = Union[int, Any]  # extensible
Shape = Sequence[DimSize]

class InconclusiveDimensionOperation(Exception):
  """Raised when we cannot conclusively compute with symbolic dimensions."""
  pass

class DimensionHandler:
  """Operations on dimension sizes.

  Dimension sizes are normally integer constants, but can also be symbolic,
  e.g., masking.Poly or jax2tf.shape_poly.DimVar.

  The base class works for integers only. Subclasses are invoked when at least
  one of the operands has a type registered in _SPECIAL_DIMENSION_HANDLERS. In
  that case, all operands are guaranteed to be either the special dimension
  type, or Python integer scalars.

  Subclasses should raise InconclusiveDimensionOperation if the result cannot
  be computed in some contexts.
  """
  def is_constant(self, d: DimSize) -> bool:
    """The dimension is a constant."""
    return True

  def symbolic_equal(self, d1: DimSize, d2: DimSize) -> bool:
    """True iff the dimension sizes are equal in all contexts; False otherwise.
    Unlike `d1 == d2` this never raises InconclusiveDimensionOperation.
    """
    return d1 == d2

  def greater_equal(self, d1: DimSize, d2: DimSize) -> bool:
    """Computes `d1 >= d2`.
    Raise InconclusiveDimensionOperation if the result is different in
    different contexts.
    """
    return d1 >= d2

  def sum(self, *ds: DimSize) -> DimSize:
    """Sum of dimensions.
    Raises InconclusiveDimensionOperation if the result cannot be represented
    by the same DimSize in all contexts.
    """
    return sum(ds)

  def diff(self, d1: DimSize, d2: DimSize) -> DimSize:
    """Difference of dimensions.
    Raises InconclusiveDimensionOperation if the result cannot be represented
    by the same DimSize in all contexts.
    """
    return d1 - d2

  def divide_shape_sizes(self, s1: Shape, s2: Shape) -> DimSize:
    """Computes integer "i" such that i  * size(s2) == size(s1).

    Raise InconclusiveDimensionOperation if there is no such integer for all
    contexts,
    """
    sz1 = int(np.prod(s1))
    sz2 = int(np.prod(s2))
    if sz1 == 0 and sz2 == 0:
      return 1
    if sz1 % sz2:
      raise InconclusiveDimensionOperation(f"Cannot divide evenly the sizes of shapes {tuple(s1)} and {tuple(s2)}")
    return sz1 // sz2

  def stride(self, d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
    """(d - window_size) // window_stride + 1"""
    return (d - window_size) // window_stride + 1

  def dilate(self, d: DimSize, dilation: int) -> DimSize:
    """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
    return 0 if d == 0 else 1 + dilation * (d - 1)

  def as_value(self, d: DimSize):
    """Turns a dimension size into a JAX value that we can compute with."""
    return d

_dimension_handler_int = DimensionHandler()
_SPECIAL_DIMENSION_HANDLERS: Dict[type, DimensionHandler] = {}

def _dim_handler_and_canonical(*dlist: DimSize) -> Tuple[DimensionHandler, Tuple[DimSize, ...]]:
  """Finds the handler for the given dimensions; also returns the canonical dimensions.

  A dimension is canonical if it is a Python integer scalar, or has a type
  registered in _SPECIAL_DIMENSION_HANDLERS.
  """
  special_handlers = set()
  canonical = []
  for d in dlist:
    handler = _SPECIAL_DIMENSION_HANDLERS.get(type(d))
    if handler:
      special_handlers.add(handler)
      canonical.append(d)
    else:
      try:
        canonical.append(operator.index(d))
      except TypeError:
        raise _invalid_shape_error(dlist)

  if len(special_handlers) > 1:
    msg = (f"Dimension size operation involves multiple special dimension types {dlist}")
    raise ValueError(msg)
  return next(iter(special_handlers), _dimension_handler_int), tuple(canonical)

def is_special_dim_size(v: Any) -> bool:
  """Checks if a value is a special DimSize."""
  handler = _SPECIAL_DIMENSION_HANDLERS.get(type(v))
  return (handler is not None)

def is_constant_dim(d: DimSize) -> bool:
  handler, ds = _dim_handler_and_canonical(d)
  return handler.is_constant(*ds)

def symbolic_equal_dim(d1: DimSize, d2: DimSize) -> bool:
  if d1 is d2: return True  # identical objects always compare equal
  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.symbolic_equal(*ds)

def symbolic_equal_one_of_dim(d1: DimSize, dlist: Sequence[DimSize]) -> bool:
  if any(d1 is d for d in dlist): return True  # identical always implies equal
  handler, ds = _dim_handler_and_canonical(d1, *dlist)
  return any([handler.symbolic_equal(ds[0], d) for d in ds[1:]])

def symbolic_equal_shape(s1: Shape, s2: Shape) -> bool:
  return (len(s1) == len(s2) and
          all(unsafe_map(symbolic_equal_dim, s1, s2)))

def greater_equal_dim(d1: DimSize, d2: DimSize) -> bool:
  # TODO(mattjj): revise this temporary workaround for dynamic shapes
  if isinstance(d1, Tracer) or isinstance(d2, Tracer):
    return True

  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.greater_equal(*ds)

def greater_equal_shape(s1: Shape, s2: Shape) -> bool:
  return all(map(greater_equal_dim, s1, s2))

def sum_dim(*ds: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(*ds)
  return handler.sum(*ds)

def sum_shapes(*ss: Shape) -> Shape:
  return tuple(map(sum_dim, *ss))

def diff_dim(d1: DimSize, d2: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(d1, d2)
  return handler.diff(*ds)

def diff_shape(s1: Shape, s2: Shape) -> Shape:
  return tuple(map(diff_dim, s1, s2))

def divide_shape_sizes(s1: Shape, s2: Shape) -> DimSize:
  """Returns an integer "i" s.t., i * size(s2) == size(s1).
  Raises if there is no such integer."""
  s1 = s1 or (1,)
  s2 = s2 or (1,)
  handler, ds = _dim_handler_and_canonical(*s1, *s2)
  return handler.divide_shape_sizes(ds[:len(s1)], ds[len(s1):])

def same_shape_sizes(s1: Shape, s2: Shape) -> bool:
  return 1 == divide_shape_sizes(s1, s2)

def is_empty_shape(s: Shape) -> bool:
  return any(symbolic_equal_dim(d, 0) for d in s)

def dilate_dim(d: DimSize, dilation: DimSize) -> DimSize:
  """Implements `0 if d == 0 else 1 + dilation * (d - 1))`"""
  handler, ds = _dim_handler_and_canonical(d, dilation)
  return handler.dilate(*ds)

def dilate_shape(s: Shape, dilations: Sequence[int]) -> Shape:
  return tuple(map(dilate_dim, s, dilations))

def stride_dim(d: DimSize, window_size: DimSize, window_stride: DimSize) -> DimSize:
  handler, ds = _dim_handler_and_canonical(d, window_size, window_stride)
  return handler.stride(*ds)

def stride_shape(s: Shape, window_size: Shape, window_stride: Shape) -> Shape:
  """(s - window_size) // window_stride + 1"""
  return tuple(map(stride_dim, s, window_size, window_stride))

def dimension_as_value(d: DimSize):
  """Turns a dimension size into a JAX value that we can compute with.
     This is the identity function for constant dimensions."""
  if isinstance(d, Tracer): return d
  handler, ds = _dim_handler_and_canonical(d)
  return handler.as_value(*ds)

def _canonicalize_dimension(dim: DimSize) -> DimSize:
  if (type(dim) in _SPECIAL_DIMENSION_HANDLERS or
      isinstance(dim, Tracer) and config.npnd_dynamic_shapes):
    return dim
  else:
    return operator.index(dim)

def canonicalize_shape(shape: Shape, context: str="") -> Shape:
  """Canonicalizes and checks for errors in a user-provided shape value.

  Args:
    shape: a Python value that represents a shape.

  Returns:
    A tuple of canonical dimension values.
  """
  try:
    return tuple(unsafe_map(_canonicalize_dimension, shape))
  except TypeError:
    pass
  raise _invalid_shape_error(shape, context)

def canonicalize_dim(d: DimSize, context: str="") -> DimSize:
  """Canonicalizes and checks for errors in a user-provided shape dimension value.

  Args:
    f: a Python value that represents a dimension.

  Returns:
    A canonical dimension value.
  """
  return canonicalize_shape((d,), context)[0]

def _invalid_shape_error(shape: Shape, context: str=""):
  msg = ("Shapes must be 1D sequences of concrete values of integer type, "
         f"got {shape}.")
  if context:
    msg += f" {context}."
  if any(isinstance(x, Tracer) and isinstance(get_aval(x), ShapedArray)
         and not isinstance(get_aval(x), ConcreteArray) for x in shape):
    msg += ("\nIf using `jit`, try using `static_argnums` or applying `jit` to "
            "smaller subfunctions.")
  return TypeError(msg)

def _check_shapelike(fun_name, arg_name, obj, non_zero_shape=False):
  """Check that `obj` is a shape-like value (e.g. tuple of nonnegative ints)."""
  if not isinstance(obj, (tuple, list, np.ndarray)):
    msg = "{} {} must be of type tuple/list/ndarray, got {}."
    raise TypeError(msg.format(fun_name, arg_name, type(obj)))
  # bool(obj) for an ndarray raises an error, so we check len
  if not len(obj):  # pylint: disable=g-explicit-length-test
    return
  if (config.npnd_dynamic_shapes and isinstance(obj, (tuple, list)) and
      any(isinstance(d, Tracer) for d in obj)):
    return  # TODO(mattjj): handle more checks in the dynamic shape case
  obj_arr = np.array(obj)
  if obj_arr.ndim != 1:
    msg = "{} {} must be rank 1, got {}."
    raise TypeError(msg.format(obj_arr.ndim))
  try:
    canonicalize_shape(obj_arr)
  except TypeError as err:
    msg = "{} {} must have every element be an integer type, got {}."
    raise TypeError(msg.format(fun_name, arg_name, tuple(map(type, obj)))) from err
  lower_bound, bound_error = (
      (1, "strictly positive") if non_zero_shape else (0, "nonnegative"))
  if not all(greater_equal_dim(d, lower_bound) for d in obj_arr):
    msg = "{} {} must have every element be {}, got {}."
    raise TypeError(msg.format(fun_name, arg_name, bound_error, obj))


class Tracer:
  pass