import numpy as np

# like np.ones, except fills with 0,1,2,3,4... instead of 1,1,1,1....
def values(shape):
  return np.arange(np.prod(shape)).reshape(shape)

def values_like(x):
  return values(np.asarray(x).shape)
