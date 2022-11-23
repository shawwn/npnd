import numpy as np

ones = np.ones
ones_like = np.ones_like
zeros = np.zeros
zeros_like = np.zeros_like

# just some helpers for generating test data.
# like np.ones, except fills with 0,1,2,3,4... instead of 1,1,1,1....
def values(shape, dtype=None) -> np.ndarray:
  return np.arange(np.prod(shape), dtype=dtype).reshape(shape)

def values_like(tensor, shape=None, dtype=None) -> np.ndarray:
  if shape is None:
    shape = np.shape(tensor)
  if dtype is None:
    dtype = np.asarray(tensor).dtype
  return values(shape, dtype=dtype)

def randi(shape, *args, dtype=None) -> np.ndarray:
  if dtype is None:
    dtype = np.int64
  if not args:
    # args = (np.prod(shape),)
    out = values(shape, dtype=dtype)
    out = out.flat[:]
    np.random.shuffle(out)
    out = out.reshape(shape)
    return out
  return np.random.randint(*args, size=shape, dtype=dtype)
