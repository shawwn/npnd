from functools import partial
from typing import Dict, Sequence
import unittest
import os
import inspect

# from absl.testing import absltest
# from absl.testing import parameterized
from npnd._src.config import flags
from npnd._src import dtypes as _dtypes

from parameterized import parameterized, param, parameterized_class

import numpy as np
import numpy.random as npr

from pytreez import tree_multimap, tree_all

FLAGS = flags.FLAGS

flags.DEFINE_integer(
  'num_generated_cases',
  int(os.getenv('NPND_NUM_GENERATED_CASES', '10')),
  help='Number of generated cases to test')


_default_tolerance = {
  _dtypes.float0: 0,
  np.dtype(np.bool_): 0,
  np.dtype(np.int8): 0,
  np.dtype(np.int16): 0,
  np.dtype(np.int32): 0,
  np.dtype(np.int64): 0,
  np.dtype(np.uint8): 0,
  np.dtype(np.uint16): 0,
  np.dtype(np.uint32): 0,
  np.dtype(np.uint64): 0,
  **({np.dtype(_dtypes.bfloat16): 1e-2} if _dtypes.bfloat16 is not None else {}),
  np.dtype(np.float16): 1e-3,
  np.dtype(np.float32): 1e-6,
  np.dtype(np.float64): 1e-15,
  np.dtype(np.complex64): 1e-6,
  np.dtype(np.complex128): 1e-15,
}

def default_tolerance():
  # if device_under_test() != "tpu":
  #   return _default_tolerance
  tol = _default_tolerance.copy()
  tol[np.dtype(np.float32)] = 1e-3
  tol[np.dtype(np.complex64)] = 1e-3
  return tol

default_gradient_tolerance = {
  **({np.dtype(_dtypes.bfloat16): 1e-1} if _dtypes.bfloat16 is not None else {}),
  np.dtype(np.float16): 1e-2,
  np.dtype(np.float32): 2e-3,
  np.dtype(np.float64): 1e-5,
  np.dtype(np.complex64): 1e-3,
  np.dtype(np.complex128): 1e-5,
}

def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=''):
  a = np.asarray(a)
  b = np.asarray(b)
  if _dtypes.bfloat16 is not None:
    a = a.astype(np.float32) if a.dtype == _dtypes.bfloat16 else a
    b = b.astype(np.float32) if b.dtype == _dtypes.bfloat16 else b
  kw = {}
  if atol: kw["atol"] = atol
  if rtol: kw["rtol"] = rtol
  with np.errstate(invalid='ignore'):
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
      # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
      # value errors. It should not do that.
      np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)
    else:
      np.testing.assert_array_equal(a, b, **kw, err_msg=err_msg)

def tolerance(dtype, tol=None):
  tol = {} if tol is None else tol
  if not isinstance(tol, dict):
    return tol
  tol = {np.dtype(key): value for key, value in tol.items()}
  dtype = _dtypes.canonicalize_dtype(np.dtype(dtype))
  return tol.get(dtype, default_tolerance()[dtype])

def _normalize_tolerance(tol):
  tol = tol or 0
  if isinstance(tol, dict):
    return {np.dtype(k): v for k, v in tol.items()}
  else:
    return {k: tol for k in _default_tolerance}

def join_tolerance(tol1, tol2):
  tol1 = _normalize_tolerance(tol1)
  tol2 = _normalize_tolerance(tol2)
  out = tol1
  for k, v in tol2.items():
    out[k] = max(v, tol1.get(k, 0))
  return out

def _assert_numpy_close(a, b, atol=None, rtol=None, err_msg=''):
  a, b = np.asarray(a), np.asarray(b)
  assert a.shape == b.shape
  atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
  rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
  _assert_numpy_allclose(a, b, atol=atol * a.size, rtol=rtol * b.size,
                         err_msg=err_msg)


def check_eq(xs, ys, err_msg=''):
  assert_close = partial(_assert_numpy_allclose, err_msg=err_msg)
  tree_all(tree_multimap(assert_close, xs, ys))

def check_close(xs, ys, atol=None, rtol=None, err_msg=''):
  assert_close = partial(_assert_numpy_close, atol=atol, rtol=rtol,
                         err_msg=err_msg)
  tree_all(tree_multimap(assert_close, xs, ys))

_CACHED_INDICES: Dict[int, Sequence[int]] = {}

def cases_from_list(xs):
  xs = list(xs)
  n = len(xs)
  k = min(n, FLAGS.num_generated_cases)
  # Random sampling for every parameterized test is expensive. Do it once and
  # cache the result.
  indices = _CACHED_INDICES.get(n)
  if indices is None:
    rng = npr.RandomState(42)
    _CACHED_INDICES[n] = indices = rng.permutation(n)
  return [xs[i] for i in indices[:k]]


def get_call_args(f, *args, **kwargs):
  sig = inspect.signature(f)
  arguments = sig.bind(*args, **kwargs).arguments
  # apply defaults:
  new_arguments = []
  for name, param in sig.parameters.items():
    try:
      new_arguments.append((name, arguments[name]))
    except KeyError:
      if param.default is not param.empty:
        val = param.default
      elif param.kind is param.VAR_POSITIONAL:
        val = ()
      elif param.kind is param.VAR_KEYWORD:
        val = {}
      else:
        continue
      new_arguments.append((name, val))
  return dict(new_arguments)

class TestCase(unittest.TestCase):
  pass

main = unittest.main

if __name__ == '__main__':
  main()
