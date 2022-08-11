import numpy as np
import math
from typing import NamedTuple, Sequence

from . import meshgrid as meshgrid_lib

def ndindex(shape, indices=()):
  if not shape:
    yield indices
  else:
    for idx in range(shape[0]):
      yield from ndindex(shape[1:], indices + (idx,))

def ndshape(indices_shape, dtype=None):
  return meshgrid_lib.meshgrid(*[np.arange(i, dtype=dtype) for i in indices_shape], indexing='ij')

def ndindices(indices, axis=None):
  indices = np.asarray(indices)
  indices_nd = list(ndshape(indices.shape, dtype=indices.dtype))
  if axis is not None:
    indices_nd[axis] = indices
  return np.stack(indices_nd, axis=-1)

# https://stackoverflow.com/a/64097936/17290907
def unstack(a, axis = 0):
  a = np.asarray(a)
  return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

def extend(a, axis, count, reverse=False):
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

def expand(a, shape, reverse=False):
  a = np.asarray(a)
  shape = tuple(shape)
  assert len(shape) == len(a.shape)
  for i in range(len(shape)):
    a = extend(a, i, shape[i] - a.shape[i], reverse=reverse)
  return a

def narrow(a, shape):
  a = np.asarray(a)
  return a[tuple([slice(0, i) for i in shape])]

def duplicate(a, count, axis):
  a = np.asarray(a)
  return np.stack([a for _ in range(count)], axis)

def get_stride_sizes(shape):
  remain_flat_size = int(np.prod(shape)) # 1 if shape is empty
  dims_to_count = []
  for dim in shape:
    remain_flat_size //= dim
    dims_to_count.append(remain_flat_size)
  return dims_to_count

def flat_inner_shape(shape, num_out_dims = 2):
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

def flat_outer_shape(shape, num_out_dims = 2):
  shape = tuple(reversed(shape))
  shape = flat_inner_shape(shape, num_out_dims=num_out_dims)
  shape = tuple(reversed(shape))
  return shape

def flat_inner_dims(tensor, num_out_dims = 2):
  tensor = np.asarray(tensor)
  shape = flat_inner_shape(tensor.shape, num_out_dims)
  return tensor.reshape(shape)

def flat_outer_dims(tensor, num_out_dims = 2):
  tensor = np.asarray(tensor)
  shape = flat_outer_shape(tensor.shape, num_out_dims)
  return tensor.reshape(shape)

def flat_nd_indices(indices, strides_shape=None, keepdims=False):
  indices = np.asarray(indices)
  if strides_shape is None:
    strides_shape = indices.shape
  strides = get_stride_sizes(strides_shape)
  indices_mat0 = flat_inner_dims(indices)
  indices_mat1 = (strides * indices_mat0)
  indices_mat = indices_mat1.sum(-1, keepdims=keepdims)
  return indices_mat

def rowcol(shape):
  if len(shape) >= 2:
    rows, cols = shape[-2:]
  else:
    rows = math.prod(shape)
    cols = None
  return rows, cols

def trigrid(N, M=None):
  if M is None:
    M = N
  rows, cols = meshgrid_lib.meshgrid(np.arange(N), np.arange(M), indexing='ij')
  return cols - rows

def tri(N, M=None, k=0, dtype=np.float):
  grid = trigrid(N, M)
  return (grid <= k).astype(dtype)
  # return (np.sum(meshgrid_lib.meshgrid(np.arange(N), np.arange(M), indexing='ij') * np.asarray([-1, 1]).reshape([-1, 1, 1]), 0) <= k).astype(dtype)

def tril(m, k=0):
  m = np.asarray(m)
  rows, cols = rowcol(m.shape)
  grid = trigrid(rows, cols)
  return (grid <= k) * m

def triu(m, k=0):
  m = np.asarray(m)
  rows, cols = rowcol(m.shape)
  grid = trigrid(rows, cols)
  return (grid >= k) * m

def trilu(v, k=0):
  v = tril(v, k=k)
  v = triu(v, k=k)
  return v

class DiagonalShapeInfo(NamedTuple):
  diagonal_size: int
  output_shape: Sequence[int]

def diagonal_shape(input_shape, offset=0, axis1=0, axis2=1):
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

def diagonal(a, offset=0, axis1=0, axis2=1):
  a = np.asarray(a)
  # Get the shape and strides of the two axes
  shape = np.shape(a)
  ndim = len(shape)
  dim1 = shape[axis1]
  dim2 = shape[axis2]
  strides = get_stride_sizes(shape)
  stride1 = strides[axis1]
  stride2 = strides[axis2]
  # Compute the data pointers and diag_size for the view
  # data = PyArray_DATA(self);
  data = 0
  # if (offset >= 0) {
  if offset >= 0:
      # offset_stride = stride2;
      offset_stride = stride2
      # dim2 -= offset;
      dim2 -= offset
  # }
  # else {
  else:
      # offset = -offset;
      offset = -offset
      # offset_stride = stride1;
      offset_stride = stride1
      # dim1 -= offset;
      dim1 -= offset
  # }
  # diag_size = dim2 < dim1 ? dim2 : dim1;
  diag_size = dim2 if dim2 < dim1 else dim1
  # if (diag_size < 0) {
  if diag_size < 0:
      # diag_size = 0;
      diag_size = 0
  # }
  # else {
  else:
      # data += offset * offset_stride;
      data += offset * offset_stride
  # }

  # Build the new shape and strides for the main data
  # i = 0;
  ret_shape = []
  ret_strides = []
  # for (idim = 0; idim < ndim; ++idim) {
  for idim in range(ndim):
      # if (idim != axis1 && idim != axis2) {
      if idim != axis1 and idim != axis2:
          # ret_shape[i] = shape[idim];
          ret_shape.append(shape[idim])
          # ret_strides[i] = strides[idim];
          ret_strides.append(strides[idim])
          # ++i;
      # }
  # }
  # ret_shape[ndim-2] = diag_size;
  ret_shape.insert(ndim-2, diag_size)
  # ret_strides[ndim-2] = stride1 + stride2;
  ret_strides.insert(ndim-2, stride1 + stride2)
  breakpoint()



def diag(v, k=0):
  v = np.asarray(v)
  rank = np.ndim(v)
  if rank not in [1, 2]:
    raise ValueError("Input must be 1- or 2-d.")
  if rank == 2:
    rows, cols = rowcol(v.shape)
    width = max(rows, cols) - min(rows, cols)
    v = tril(v, k=k)
    v = triu(v, k=k)
    v = v.sum(1)
    print(k, width, cols, rows, v.shape)
    k -= cols - rows
    if k < 0:
      v = v[:k]
    else:
      v = v[k:]
    return v
  else:
    raise NotImplementedError

