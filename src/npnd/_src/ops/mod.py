import numpy as np

def gettype(kind, subtype, dtype):
  dtype = np.dtype(dtype)
  if np.issubdtype(dtype, subtype):
    return dtype
  else:
    return np.dtype(f'{kind}{dtype.itemsize}')

def inttype(dtype: np.dtype):
  return gettype('i', np.integer, dtype)

def floattype(dtype: np.dtype):
  return gettype('f', np.float, dtype)

def itrunc(x):
  x = np.asanyarray(x)
  return x.astype(inttype(x.dtype))

def trunc(x):
  return sign(x) * floor(abs(x))

def sign(x):
  x = np.asanyarray(x)
  out = (x > 0).astype(int) - (x < 0).astype(int)
  return out.astype(x.dtype)

def iceil(x):
  ix = itrunc(x)
  return np.where((x == ix) | (x < 0), ix, ix + np.array(1, dtype=ix.dtype))

def ifloor(x):
  ix = itrunc(x)
  return np.where((x == ix) | (x > 0), ix, ix - np.array(1, dtype=ix.dtype))

def round(x):
  x = np.asanyarray(x)
  return floor(x + 0.5).astype(x.dtype)

def ceil(x):
  x = np.asanyarray(x)
  return iceil(x).astype(floattype(x.dtype))

def floor(x):
  x = np.asanyarray(x)
  return ifloor(x).astype(floattype(x.dtype))

def fmod(a, n):
  a = np.asanyarray(a)
  n = np.asanyarray(n)
  return a - itrunc(a / n) * n

def mod(a, n):
  if True:
    a = np.asanyarray(a)
    n = np.asanyarray(n)
    return a - ifloor(a / n) * n
  else:
    # correct:
    val = fmod(a, n)
    return np.where(val != 0, np.where(np.sign(a) != np.sign(n), val + n, val), val)


def equal(f1, f2, *args, **kws):
  x1 = np.asanyarray(f1(*args, **kws))
  x2 = np.asanyarray(f2(*args, **kws));
  close = np.allclose(x1, x2)
  types = x1.dtype == x2.dtype
  assert close, f"{x1}, {x2}"
  assert types
  return close and types

def unzip2(xys):
  xs = []
  ys = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)

a, b = unzip2([(a, b) for a in np.arange(-7, 7) for b in np.arange(-7, 7) if b != 0])
a2, b2 = unzip2([(a, b) for a in np.arange(-2, 2, 0.25) for b in np.arange(-2, 2, 0.25) if b != 0])
a = np.asarray(a)
b = np.asarray(b)
a2 = np.asarray(a2)
b2 = np.asarray(b2)
x = np.random.random(size=(100,)) * 5 - 2.5
y = np.random.random(size=(100,)) * 5 - 2.5

cases = [(a, b), (a2, b2), (x, y)]
#cases += [(np.asarray(-3).astype(np.uint64), np.asarray(3).astype(np.uint64))]
# cases += [(np.asarray([-3]).astype(np.uint64), np.asarray([3]).astype(np.uint64))]

for i, args in enumerate(cases):
  assert equal(np.mod, mod, *args)
  assert equal(np.fmod, fmod, *args)
  for j, arg in enumerate(list(args) + [x.astype(np.uint64) for x in args]):
    # if j < len(args):
    #   assert equal(np.round, round, arg) # sign of 0.0 mismatch
    assert equal(np.trunc, trunc, arg)
    assert equal(np.floor, floor, arg)
    assert equal(np.ceil, ceil, arg)
    assert equal(np.sign, sign, arg)
    assert np.issubdtype(itrunc(arg).dtype, np.integer)
    assert np.issubdtype(ifloor(arg).dtype, np.integer)
    assert np.issubdtype(iceil(arg).dtype, np.integer)
    if np.issubdtype(arg.dtype, np.integer):
      assert ifloor(arg).dtype == arg.dtype
      assert iceil(arg).dtype == arg.dtype
      assert itrunc(arg).dtype == arg.dtype
