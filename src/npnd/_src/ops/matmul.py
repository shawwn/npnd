from . import broadcast as broadcast_lib

def get_dim(shape, i):
  if i < 0:
    i += len(shape)
  if i < 0:
    return 0
  if i >= len(shape):
    return len(shape) - 1
  return i

# >>> (np.ones((2,3,5,2,)) @ np.ones((2,3,2,8))).shape
# (2, 3, 5, 8)
# >>> # signature is (m?,n),(n,p?)->(m?,p?)
def matmul_shape(shape1, shape2):
  i1 = get_dim(shape1, -1)
  i2 = get_dim(shape2, -2)
  if shape1[i1] != shape2[i2]:
    raise ValueError(f"matmul input operand 1 has a mismatch in its core dimension 0 (size {shape1[i1]} is different from {shape2[i2]})")
  m = shape1[i1-1:i1]
  p = shape2[i2+1:]
  batch_a = shape1[:i1-1]
  batch_b = shape2[:i2]
  batch = broadcast_lib.broadcast_shape(batch_a, batch_b)
  return batch + m + p

