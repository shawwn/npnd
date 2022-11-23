import numpy as np
import math
from typing import NamedTuple, Sequence, Tuple, List, Optional

from . import meshgrid as meshgrid_lib

Shape = Sequence[int]

def shape(a):
  return np.shape(a)

def ndim(a):
  return len(shape(a))


class flatiter:
  def __init__(self, a):
    self.a = np.asanyarray(a)

def ndcoord(shape: Shape, index: int):
  if index < 0:
    index += math.prod(shape)
  if index >= math.prod(shape):
    raise IndexError("ndcoord index out of range")
  coord = []
  stride = math.prod(shape)
  for dim in shape:
    stride //= dim
    coord.append(index // stride)
    index %= stride
  return tuple(coord)

def ndindex(shape: Shape, indices=()) -> Shape:
  if not shape:
    yield indices
  else:
    for idx in range(shape[0]):
      yield from ndindex(shape[1:], indices + (idx,))

def ndindex(shape: Shape, dim: int = 0, indices = (), out = None):
  if out is None:
    out = []
  if dim == len(shape):
    out.append(indices)
  else:
    for idx in range(shape[dim]):
      ndindex(shape, dim + 1, indices + (idx,), out)
  return out

def ndindex(shape: Shape):
  out = []
  def iterate(dim: int = 0, indices = ()):
    if dim == len(shape):
      out.append(indices)
    else:
      for idx in range(shape[dim]):
        iterate(dim + 1, indices + (idx,))
  iterate()
  return out

def ndindex_dim(shape: Shape, dim: int, indices: Shape, out: List):
  if dim == len(shape):
    out.append(indices)
  else:
    for idx in range(shape[dim]):
      ndindex_dim(shape, dim + 1, indices + (idx,), out)
  return out

def ndindex(shape: Shape):
   return ndindex_dim(shape, 0, (), [])

def ndindex(shape: Shape):
  coords = []
  numel = math.prod(shape)
  for index in range(numel):
    coord = []
    stride = numel
    for dim in shape:
      stride //= dim
      coord.append(index // stride)
      index %= stride
    coords.append(tuple(coord))
  return coords

def ndcoords(shape: Shape):
  return np.asarray(list(ndindex(shape))).reshape(tuple(shape) + (len(shape),))

def ndshape(indices_shape: Shape, dtype=None, broadcast=True) -> List[np.ndarray]:
  return meshgrid_lib.meshgrid(*[np.arange(i, dtype=dtype) for i in indices_shape], indexing='ij', broadcast=broadcast)

def ndaxis(indices_shape: Shape, axis: int, dtype=None) -> List[np.ndarray]:
  return meshgrid_lib.meshgrid(*[np.arange(i, dtype=dtype) for i in indices_shape], indexing='ij')

def ndindices(indices, axis=None, shape=None) -> np.ndarray:
  indices = np.asarray(indices)
  if shape is None:
    shape = indices.shape
  indices_nd = list(ndshape(shape, dtype=indices.dtype))
  if axis is not None:
    indices_nd[axis] = indices
  return np.stack(indices_nd, axis=-1)

def iota(shape: Shape, iota_dimension: int, dtype=None):
  return ndshape(shape, dtype=dtype)[iota_dimension]

# https://stackoverflow.com/a/64097936/17290907
def unstack(a, axis = 0, keepdims=False) -> List[np.ndarray]:
  a = np.asarray(a)
  return [np.squeeze(e, axis) if not keepdims else e for e in np.split(a, a.shape[axis], axis = axis)]

def stack(ls: Tuple[np.ndarray, ...], axis=0) -> np.ndarray:
  ls2 = [np.expand_dims(l, axis=axis) for l in ls]
  return concat(ls2, axis=axis)

def concat(ls, axis=0):
  return np.concatenate(ls, axis=axis)

def extend(a, axis, count, reverse=False) -> np.ndarray:
  a = np.asarray(a)
  if count <= 0:
    return a
  shape = list(np.shape(a))
  shape[axis] = count
  padding = np.zeros_like(a, shape=shape)
  if reverse:
    return np.concatenate([padding, a], axis=axis)
  else:
    return np.concatenate([a, padding], axis=axis)

def expand(a, shape: Shape, reverse=False) -> np.ndarray:
  a = np.asarray(a)
  shape = tuple(shape)
  assert len(shape) == len(a.shape)
  for i in range(len(shape)):
    a = extend(a, i, shape[i] - a.shape[i], reverse=reverse)
  return a

def narrow(a, shape: Shape) -> np.ndarray:
  a = np.asarray(a)
  return a[tuple([slice(0, i) for i in shape])]

def duplicate(a, count, axis):
  a = np.asarray(a)
  return np.stack([a for _ in range(count)], axis)

def get_stride_sizes(shape: Shape) -> Shape:
  remain_flat_size = int(np.prod(shape)) # 1 if shape is empty
  dims_to_count = []
  for dim in shape:
    remain_flat_size //= dim
    dims_to_count.append(remain_flat_size)
  return tuple(dims_to_count)

def flat_inner_shape(shape: Shape, num_out_dims = 2) -> Shape:
  assert num_out_dims > 0
  out_dims = [0 for i in range(num_out_dims)]
  offset = len(shape) - num_out_dims
  for out_dim in reversed(range(num_out_dims)):
    in_dim = out_dim + offset
    out_dims[out_dim] = 1 if in_dim < 0 else shape[in_dim]
  for in_dim in range(offset):
    out_dims[0] *= shape[in_dim]
  return tuple(out_dims)

assert flat_inner_shape((4,4,4)) == (16, 4)
assert flat_inner_shape((4,4,4), 1) == (64,)
assert flat_inner_shape((4,4,4), 2) == (16, 4)
assert flat_inner_shape((4,4,4), 3) == (4, 4, 4)
assert flat_inner_shape((4,4,4), 4) == (1, 4, 4, 4)
assert flat_inner_shape((4,4,4), 5) == (1, 1, 4, 4, 4)

assert flat_inner_shape((4,)) == (1, 4)
assert flat_inner_shape((4,), 1) == (4,)
assert flat_inner_shape((4,), 2) == (1, 4)
assert flat_inner_shape((4,), 3) == (1, 1, 4)

def flat_outer_shape(shape: Shape, num_out_dims = 2) -> Shape:
  shape = tuple(reversed(shape))
  shape = flat_inner_shape(shape, num_out_dims=num_out_dims)
  shape = tuple(reversed(shape))
  return shape

def flat_inner_dims(tensor, num_out_dims = 2) -> np.ndarray:
  tensor = np.asarray(tensor)
  shape = flat_inner_shape(tensor.shape, num_out_dims)
  return tensor.reshape(shape)

def flat_outer_dims(tensor, num_out_dims = 2) -> np.ndarray:
  tensor = np.asarray(tensor)
  shape = flat_outer_shape(tensor.shape, num_out_dims)
  return tensor.reshape(shape)

def flat_nd_indices(indices, strides_shape: Shape = None, keepdims=False) -> np.ndarray:
  indices = np.asarray(indices)
  if strides_shape is None:
    strides_shape = indices.shape
  strides = get_stride_sizes(strides_shape)
  indices_mat0 = flat_inner_dims(indices)
  indices_mat1 = (strides * indices_mat0)
  indices_mat = indices_mat1.sum(-1, keepdims=keepdims)
  return indices_mat

def rowcol(shape: Shape) -> Tuple[int, Optional[int]]:
  if len(shape) >= 2:
    rows, cols = shape[-2:]
  else:
    rows = math.prod(shape)
    cols = None
  return rows, cols

def trigrid(N: int, M: Optional[int] = None) -> np.ndarray:
  if M is None:
    M = N
  rows, cols = meshgrid_lib.meshgrid(np.arange(N), np.arange(M), indexing='ij')
  return cols - rows

def tri(N: int, M: Optional[int] = None, k=0, dtype=np.float) -> np.ndarray:
  grid = trigrid(N, M)
  return (grid <= k).astype(dtype)
  # return (np.sum(meshgrid_lib.meshgrid(np.arange(N), np.arange(M), indexing='ij') * np.asarray([-1, 1]).reshape([-1, 1, 1]), 0) <= k).astype(dtype)

def tril(m, k=0, off_value=0) -> np.ndarray:
  m = np.asarray(m)
  rows, cols = rowcol(m.shape)
  grid = trigrid(rows, cols)
  #return (grid <= k) * m
  return np.where(grid <= k, m, off_value)

def triu(m, k=0, off_value=0) -> np.ndarray:
  m = np.asarray(m)
  rows, cols = rowcol(m.shape)
  grid = trigrid(rows, cols)
  # return (grid >= k) * m
  return np.where(grid >= k, m, off_value)

def trilu(v, k=0, off_value=0) -> np.ndarray:
  v = tril(v, k=k, off_value=off_value)
  v = triu(v, k=k, off_value=off_value)
  return v

def diagonal_shape(input_shape: Shape, offset=0, axis1=0, axis2=1) -> Tuple[int, Shape, Shape]:
  shape = tuple(input_shape)
  ndim = len(shape)
  dim1 = shape[axis1]
  dim2 = shape[axis2]
  strides = get_stride_sizes(shape)
  stride1 = strides[axis1]
  stride2 = strides[axis2]

  data = 0
  if offset >= 0:
    offset_stride = stride2
    dim2 -= offset
  else:
    offset = -offset
    offset_stride = stride1
    dim1 -= offset

  diag_size = min(dim1, dim2)
  if diag_size < 0:
    diag_size = 0
  else:
    data += offset * offset_stride

  # Build the new shape and strides for the main data
  # i = 0;
  output_shape = []
  output_strides = []
  # for (idim = 0; idim < ndim; ++idim) {
  for idim in range(ndim):
    # if (idim != axis1 && idim != axis2) {
    if idim != axis1 and idim != axis2:
      # output_shape[i] = shape[idim];
      output_shape.append(shape[idim])
      # output_strides[i] = strides[idim];
      output_strides.append(strides[idim])
      # ++i;
    # }
  # }
  # output_shape[ndim-2] = diag_size;
  output_shape.insert(ndim-2, diag_size)
  # output_strides[ndim-2] = stride1 + stride2;
  output_strides.insert(ndim-2, stride1 + stride2)
  return diag_size, output_shape, output_strides

def diagonal_shape(input_shape: Shape, offset=0, axis1=0, axis2=1) -> Shape:
  if axis1 < 0:
    axis1 += len(input_shape)
  if axis2 < 0:
    axis2 += len(input_shape)
  if axis1 == axis2:
    raise ValueError("axis1 and axis2 cannot be the same")
  dim1 = input_shape[axis1]
  dim2 = input_shape[axis2]
  if offset >= 0:
    dim2 -= offset
  else:
    dim1 += offset
  output_shape = []
  for idim in range(len(input_shape)):
    if idim not in [axis1, axis2]:
      output_shape.append(input_shape[idim])
  diagonal_size = min(dim1, dim2)
  output_shape.append(diagonal_size)
  return tuple(output_shape)

def diagonal(v, offset=0, axis1=0, axis2=1) -> np.ndarray:
  v = np.asanyarray(v)
  output_shape = diagonal_shape(v.shape, offset=offset, axis1=axis1, axis2=axis2)
  if axis1 < 0:
    axis1 += len(v.shape)
  if axis2 < 0:
    axis2 += len(v.shape)
  indices = [idx for idx in ndindex(v.shape) if idx[axis1] + min(0,offset) == idx[axis2] - max(0,offset)]
  indices.sort(key=lambda idx: idx[axis1])
  indices.sort(key=lambda idx: idx[axis2])
  for idim in reversed(range(len(v.shape))):
    if idim not in [axis1, axis2]:
      indices.sort(key=lambda idx: idx[idim])
  if len(indices) <= 0:
    out = np.array([], dtype=v.dtype)
  else:
    from . import gather_nd as gather_nd_lib
    out = gather_nd_lib.gather_nd(v, indices)
  out = out.reshape(output_shape)
  return out

def diag(v, k=0) -> np.ndarray:
  v = np.asarray(v)
  rank = np.ndim(v)
  if rank not in [1, 2]:
    raise ValueError("Input must be 1- or 2-d.")
  if rank == 2:
    width, output_shape, output_strides = diagonal_shape(v.shape, offset=k)
    v = trilu(v, k=k)
    v = v.sum(1)
    if k >= 0:
      v = v[:width]
    else:
      v = v[-width:]
    return v
  else:
    shape = v.shape
    v = np.reshape(v, shape + (1,))
    v = np.broadcast_to(v, shape + shape)
    v = trilu(v, k=k)
    return v

def eye(N, M=None, k=0, dtype=float):
  if M is None:
    M = N
  out = np.ones((N, M), dtype=dtype)
  out = trilu(out, k=k)
  return out

def identity(n, dtype=float):
  return eye(n, n, dtype=dtype)