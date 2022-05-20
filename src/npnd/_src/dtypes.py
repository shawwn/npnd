# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Array type functions.
#
# NPND dtypes differ from NumPy in both:
# a) their type promotion rules, and
# b) the set of supported types (e.g., bfloat16),
# so we need our own implementation that deviates from NumPy in places.


import functools
from typing import Any, Dict, Optional

import numpy as np


from npnd._src.config import flags, config

# from npnd._src import traceback_util
# traceback_util.register_exclusion(__file__)

FLAGS = flags.FLAGS

# bfloat16 support
try:
  raise ImportError # TODO: Make this configurable?
  from jax._src.lib import xla_client
  bfloat16: Optional[type] = xla_client.bfloat16
  _bfloat16_dtype: Optional[np.dtype] = np.dtype(bfloat16)
except ImportError:
  #bfloat16 = np.float32 # Just punt and use float32 for now.
  bfloat16: Optional[type] = None
  _bfloat16_dtype: Optional[np.dtype] = np.dtype([('bfloat16', '<V2', 2)])

# Default types.
bool_: type = np.bool_
int_: type = np.int32 if config.npnd_default_dtype_bits == '32' else np.int64
uint: type = np.uint32 if config.npnd_default_dtype_bits == '32' else np.uint64
float_: type = np.float32 if config.npnd_default_dtype_bits == '32' else np.float64
complex_: type = np.complex64 if config.npnd_default_dtype_bits == '32' else np.complex128
_default_types = {'b': bool_, 'i': int_, 'u': uint, 'f': float_, 'c': complex_}

# Trivial vectorspace datatype needed for tangent values of int/bool primals
float0: np.dtype = np.dtype([('float0', np.void, 0)])

_dtype_to_32bit_dtype = {
    np.dtype('int64'): np.dtype('int32'),
    np.dtype('uint64'): np.dtype('uint32'),
    np.dtype('float64'): np.dtype('float32'),
    np.dtype('complex128'): np.dtype('complex64'),
}

@functools.lru_cache(maxsize=None)
def _canonicalize_dtype(x64_enabled, dtype):
  """Convert from a dtype to a canonical dtype based on config.x64_enabled."""
  try:
    dtype = np.dtype(dtype)
  except TypeError as e:
    raise TypeError(f'dtype {dtype!r} not understood') from e

  if x64_enabled:
    return dtype
  else:
    return _dtype_to_32bit_dtype.get(dtype, dtype)

def canonicalize_dtype(dtype):
  return _canonicalize_dtype(config.x64_enabled, dtype)

# Default dtypes corresponding to Python scalars.
python_scalar_dtypes : dict = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}

def scalar_type_of(x):
  typ = dtype(x)
  if bfloat16 is not None:
    if typ == bfloat16:
      return float
  elif np.issubdtype(typ, np.bool_):
    return bool
  elif np.issubdtype(typ, np.integer):
    return int
  elif np.issubdtype(typ, np.floating):
    return float
  elif np.issubdtype(typ, np.complexfloating):
    return complex
  else:
    raise TypeError("Invalid scalar value {}".format(x))


def _scalar_type_to_dtype(typ: type, value: Any = None):
  """Return the numpy dtype for the given scalar type.

  Raises
  ------
  OverflowError: if `typ` is `int` and the value is too large for int64.

  Examples
  --------
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(float)
  dtype('float32')
  >>> _scalar_type_to_dtype(complex)
  dtype('complex64')
  >>> _scalar_type_to_dtype(int)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 0)
  dtype('int32')
  >>> _scalar_type_to_dtype(int, 1 << 63)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
  OverflowError: Python int 9223372036854775808 too large to convert to int32
  """
  dtype = canonicalize_dtype(python_scalar_dtypes[typ])
  if typ is int and value is not None:
    if value < np.iinfo(dtype).min or value > np.iinfo(dtype).max:
      raise OverflowError(f"Python int {value} too large to convert to {dtype}")
  return dtype


def coerce_to_array(x, dtype=None):
  """Coerces a scalar or NumPy array to an np.array.

  Handles Python scalar type promotion according to NPND's rules, not NumPy's
  rules.
  """
  if dtype is None and type(x) in python_scalar_dtypes:
    dtype = _scalar_type_to_dtype(type(x), x)
  return np.asarray(x, dtype)

iinfo = np.iinfo

class _Bfloat16MachArLike:
  def __init__(self):
    smallest_normal = float.fromhex("0x1p-126")
    self.smallest_normal = bfloat16(smallest_normal)

class finfo(np.finfo):
  __doc__ = np.finfo.__doc__
  _finfo_cache: Dict[np.dtype, np.finfo] = {}
  @staticmethod
  def _bfloat16_finfo():
    def float_to_str(f):
      return "%12.4e" % float(f)

    bfloat16 = _bfloat16_dtype.type
    tiny = float.fromhex("0x1p-126")
    resolution = 0.01
    eps = float.fromhex("0x1p-7")
    epsneg = float.fromhex("0x1p-8")
    max = float.fromhex("0x1.FEp127")

    obj = object.__new__(np.finfo)
    obj.dtype = _bfloat16_dtype
    obj.bits = 16
    obj.eps = bfloat16(eps)
    obj.epsneg = bfloat16(epsneg)
    obj.machep = -7
    obj.negep = -8
    obj.max = bfloat16(max)
    obj.min = bfloat16(-max)
    obj.nexp = 8
    obj.nmant = 7
    obj.iexp = obj.nexp
    obj.precision = 2
    obj.resolution = bfloat16(resolution)
    obj._machar = _Bfloat16MachArLike()
    if not hasattr(obj, "tiny"):
      obj.tiny = bfloat16(tiny)

    obj._str_tiny = float_to_str(tiny)
    obj._str_smallest_normal = float_to_str(tiny)
    obj._str_max = float_to_str(max)
    obj._str_epsneg = float_to_str(epsneg)
    obj._str_eps = float_to_str(eps)
    obj._str_resolution = float_to_str(resolution)
    return obj

  def __new__(cls, dtype):
    if isinstance(dtype, str) and dtype == 'bfloat16' or dtype == _bfloat16_dtype:
      if _bfloat16_dtype not in cls._finfo_cache:
        cls._finfo_cache[_bfloat16_dtype] = cls._bfloat16_finfo()
      return cls._finfo_cache[_bfloat16_dtype]
    return super().__new__(cls, dtype)


def _issubclass(a, b):
  """Determines if ``a`` is a subclass of ``b``.

  Similar to issubclass, but returns False instead of an exception if `a` is not
  a class.
  """
  try:
    return issubclass(a, b)
  except TypeError:
    return False

def issubdtype(a, b):
  if bfloat16 is not None:
    if a == "bfloat16":
      a = bfloat16
    if a == bfloat16:
      if isinstance(b, np.dtype):
        return b == _bfloat16_dtype
      else:
        return b in [bfloat16, np.floating, np.inexact, np.number]
  if not _issubclass(b, np.generic):
    # Workaround for NPND scalar types. NumPy's issubdtype has a backward
    # compatibility behavior for the second argument of issubdtype that
    # interacts badly with NPND's custom scalar types. As a workaround,
    # explicitly cast the second argument to a NumPy type object.
    b = np.dtype(b).type
  return np.issubdtype(a, b)

can_cast = np.can_cast
issubsctype = np.issubsctype

# Enumeration of all valid NPND types in order.
_weak_types = [int, float, complex]
_npnd_types = [
  np.dtype('bool'),
  np.dtype('uint8'),
  np.dtype('uint16'),
  np.dtype('uint32'),
  np.dtype('uint64'),
  np.dtype('int8'),
  np.dtype('int16'),
  np.dtype('int32'),
  np.dtype('int64'),
  *([np.dtype(bfloat16)] if bfloat16 is not None else []),
  np.dtype('float16'),
  np.dtype('float32'),
  np.dtype('float64'),
  np.dtype('complex64'),
  np.dtype('complex128'),
]
_npnd_dtype_set = set(_npnd_types) | {float0}

def dtype(x, *, canonicalize=False):
  """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  elif isinstance(x, type) and x in python_scalar_dtypes:
    dt = python_scalar_dtypes[x]
  elif type(x) in python_scalar_dtypes:
    dt = python_scalar_dtypes[type(x)]
  else:
    dt = np.result_type(x)
  if dt not in _npnd_dtype_set:
    raise TypeError(f"Value '{x}' with dtype {dt} is not a valid NPND array "
                    "type. Only arrays of numeric types are supported by NPND.")
  return canonicalize_dtype(dt) if canonicalize else dt
