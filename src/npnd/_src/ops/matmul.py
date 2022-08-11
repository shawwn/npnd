import numpy as np
from functools import reduce
import operator as op

from . import broadcast as broadcast_lib
from . import meshgrid as meshgrid_lib
from . import gather_nd as gather_nd_lib
from . import shape as shape_lib
from . import reshape as reshape_lib

def get_dim(shape, i):
  if i < 0:
    i += len(shape)
  if i < 0:
    return 0
  if i >= len(shape):
    return len(shape) - 1
  return i

def broadcast_ranks(shape1, shape2, shape1_insert_pos=-1, shape2_insert_pos=0):
  shape1 = list(shape1)
  shape2 = list(shape2)
  if shape1_insert_pos < 0:
    shape1_insert_pos += len(shape1) + 1
  if shape2_insert_pos < 0:
    shape2_insert_pos += len(shape2) + 1
  while len(shape1) < len(shape2):
    shape1.insert(shape1_insert_pos, 1)
  while len(shape2) < len(shape1):
    shape2.insert(shape2_insert_pos, 1)
  return broadcast_lib.broadcast_shapes(shape1, shape2)

# >>> (np.ones((2,3,5,2,)) @ np.ones((2,3,2,8))).shape
# (2, 3, 5, 8)
# >>> # signature is (m?,n),(n,p?)->(m?,p?)
def matmul_shape(shape1, shape2, shape1_dim = -1, shape2_dim = -2):
  i1 = get_dim(shape1, shape1_dim)
  i2 = get_dim(shape2, shape2_dim)
  if shape1[i1] != shape2[i2]:
    raise ValueError(f"matmul input operand 1 has a mismatch in its core dimension 0 (size {shape1[i1]} is different from {shape2[i2]})")
  shape1 = list(shape1)
  shape2 = list(shape2)
  shape1_batch, m, n1 = shape1[:i1], shape1[i1:i1+1], shape1[i1+1:]
  shape2_batch, n2, p = shape2[:i2+1], shape2[i2:i2+1], shape2[i2+1:]
  # shape1[i1] = 1
  # shape2[i2] = 1
  # shape = broadcast_lib.broadcast_shape(shape1, shape2)
  # shape_lh = broadcast_lib.broadcast_shape(shape1_lh, shape2_lh)
  # shape_rh = broadcast_lib.broadcast_shape(shape1_rh, shape2_rh)
  breakpoint()
  shape_lh = broadcast_ranks(shape1_lh, shape2_lh)
  shape_rh = broadcast_ranks(shape1_rh, shape2_rh)
  shape = shape_lh + shape_rh
  exclude = []
  if len(shape1) == 1 or len(shape2) == 1:
    if shape[i1] == 1:
      exclude.append(i1)
    if shape[i2] == 1:
      exclude.append(i2)
  shape = tuple(dim for idim, dim in enumerate(shape) if idim not in exclude)
  return shape

def matmul_shape(shape1, shape2):
  shape1 = tuple(shape1)
  shape2 = tuple(shape2)
  i1 = get_dim(shape1, -1)
  i2 = get_dim(shape2, -2)
  if shape1[i1] != shape2[i2]:
    raise ValueError(f"matmul input operand 1 has a mismatch in its core dimension 0 (size {shape1[i1]} is different from {shape2[i2]})")
  m = shape1[i1-1:i1]
  p = shape2[i2+1:]
  batch_a = shape1[:i1-1]
  batch_b = shape2[:i2]
  batch = np.broadcast_shapes(batch_a, batch_b)
  return batch + m + p

def matmul_shape(shape1, shape2, x_dim = -1, y_dim = -2):
  shape1 = tuple(shape1)
  shape2 = tuple(shape2)
  i1 = get_dim(shape1, x_dim)
  i2 = get_dim(shape2, y_dim)
  if shape1[i1] != shape2[i2]:
    raise ValueError(f"matmul input operand 1 has a mismatch in its core dimension 0 (size {shape1[i1]} is different from {shape2[i2]})")
  # m = shape1[i2-1:i2]
  # p = shape2[i1+1:]
  m = shape1[i1-1:i1]
  p = shape2[i2+1:]
  batch_a = shape1[:min(i1, i2)]
  batch_b = shape2[:min(i1, i2)]
  batch = np.broadcast_shapes(batch_a, batch_b)
  return batch + m + p

def matmul(x, y, x_dim = -1, y_dim = -2):
  x = np.asarray(x)
  y = np.asarray(y)
  # shape = matmul_shape(x.shape, y.shape, x_dim, y_dim)
  def contract(x_indices_nd, y_indices_nd):
    X = gather_nd_lib.gather_nd(x, x_indices_nd)
    X = reshape_lib.expand_dims(X, x_dim)
    Y = gather_nd_lib.gather_nd(y, y_indices_nd)
    Y = reshape_lib.expand_dims(Y, y_dim)
    return X * Y
  xs_nd = shape_lib.unstack(meshgrid_lib.ndcoords(x.shape), get_dim(x.shape, x_dim))
  ys_nd = shape_lib.unstack(meshgrid_lib.ndcoords(y.shape), get_dim(y.shape, y_dim))
  # outs = inner(x_nd, y_nd, contract)
  # out = np.sum(outs, 0)
  #out = np.zeros(shape, dtype=x.dtype)
  out = None
  for x_nd, y_nd in zip(xs_nd, ys_nd):
    z = contract(x_nd, y_nd)
    if out is None:
      out = z
    else:
      out += z
    #out = np.sum([out, contract(x_nd, y_nd)], axis=0)
  if out is not None:
    return out.reshape(shape)

def matmul(x, y, x_dim = -1, y_dim = -2):
  x = np.asarray(x)
  y = np.asarray(y)
  shape = matmul_shape(x.shape, y.shape, x_dim, y_dim)
  x_dim = get_dim(x.shape, x_dim)
  y_dim = get_dim(y.shape, y_dim)
  out = None
  dim1 = x.shape[x_dim]
  dim2 = y.shape[y_dim]
  assert dim1 == dim2
  for coord in range(dim1):
    X = gather_nd_lib.take(x, [coord], x_dim)
    X = reshape_lib.expand_dims(X, -1)
    # X = reshape_lib.expand_dims(X, get_dim(x.shape, x_dim - 1))
    Y = gather_nd_lib.take(y, [coord], y_dim)
    Y = reshape_lib.expand_dims(Y, 0)
    # Y = reshape_lib.expand_dims(Y, get_dim(y.shape, y_dim - 1))
    XY = X * Y
    if out is None:
      out = XY
    else:
      out += XY
  return out.reshape(shape)

def inner(x, y, fn=lambda a, b: (a,b)):
  return [fn(a, b) for a, b in zip(x, y)]

def outer(x, y, fn=lambda a, b: (a,b)):
  return [[fn(a, b) for b in y] for a in x]

def explode(x, y, xdims, ydims):
  print(x.shape, y.shape, xdims, ydims)
  #return (np.expand_dims(x, -1) * np.expand_dims(y, 0)) if not xdims or not ydims else np.sum([explode(a, b, xdims[1:], ydims[1:]) for a, b in zip(npnd.unstack(x, xdims[0], keepdims=True), npnd.unstack(y, ydims[0], keepdims=True))], axis=0)
  if not xdims or not ydims:
    #return np.expand_dims(x, -1) * np.expand_dims(y, 0)
    return x * y
  else:
    lh = shape_lib.unstack(x, xdims[0], keepdims=True)
    rh = shape_lib.unstack(y, ydims[0], keepdims=True)
    sums = [explode(a, b, xdims[1:], ydims[1:]) for a, b in zip(lh, rh)]
    return np.sum(sums, axis=0)

def uniq(xs):
  ys = []
  for x in xs:
    if x not in ys:
      ys.append(x)
  return type(xs)(ys)

def maybe_squeeze(x, axes):
  x = np.asarray(x)
  axes = reshape_lib.normalize_axis_tuple(axes, len(x.shape), allow_duplicate=True)
  axes = uniq(axes)
  axes = tuple(ax for ax in axes if x.shape[ax] == 1)
  out = np.squeeze(x, axes)
  return out

def matmul(x, y, xdims=(-1,), ydims=(-2,)):
  xdims = reshape_lib.normalize_axis_tuple(xdims, len(np.shape(x)))
  ydims = reshape_lib.normalize_axis_tuple(ydims, len(np.shape(y)))
  out = explode(x, y, xdims, ydims)
  out = maybe_squeeze(out, xdims + ydims)
  return out

def einsum(*operands):
  if len(operands) == 2:
    op, sublistout = operands
    sublist = list(range(len(np.shape(op))))
    return einsum(op, sublist, sublistout)
  if isinstance(operands[0], str):
    raise NotImplementedError


def explode2(values, dims):
  #return (np.expand_dims(x, -1) * np.expand_dims(y, 0)) if not xdims or not ydims else np.sum([explode(a, b, xdims[1:], ydims[1:]) for a, b in zip(npnd.unstack(x, xdims[0], keepdims=True), npnd.unstack(y, ydims[0], keepdims=True))], axis=0)
  if any([not x for x in dims]):
    #return [np.expand_dims(v, 0) for i, v in enumerate(values)]
    print(len(values), [np.shape(x) for x in values])
    return values
  else:
    # for xy_dims in zip(*dims):
    #   vs = [shape_lib.unstack(v, dim[0], keepdims=True) for v, dim in zip(values, xy_dims)
    #   lh = shape_lib.unstack(x, xdims[0], keepdims=True)
    #   rh = shape_lib.unstack(y, ydims[0], keepdims=True)
    #   sums = [explode(a, b, xdims[1:], ydims[1:]) for a, b in zip(lh, rh)]

    axis = [axes[0] for axes in dims]
    dimz = [axes[1:] for axes in dims]
    def merge(vals):
      if not isinstance(vals[0], np.ndarray):
        vals = list(reduce(op.concat, vals))
      return reduce(op.add, vals)
      # else:
      # return broadcast_lib.broadcast_arrays(*arrays)
    valz = [merge(explode2(shape_lib.unstack(val, ax, keepdims=True), dimz)) for val, ax in zip(values, axis)]
    #dimz = [axes[1:] for axes in dims]
    #sums = [explode2(
    #sums = [explode2(shape_lib.unstack(val, axes[0], keepdims=True), axes[1:]) for val, axes in zip(values, dims)]
    #return np.sum(valz, axis=0)
    return valz
