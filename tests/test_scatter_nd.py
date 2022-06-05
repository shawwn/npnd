import numpy as np

import npnd
from npnd import scatter_nd
from npnd import scatter
import npnd.test_util as ntu
import npnd.numpy as nnp
import functools
import inspect

# reference implementation of scatter_nd.

def scatter_nd_ref(original, indices, values, reduction=None):
  assert reduction in [None, 'add', 'mul'], "Unknown reduction type"
  original = np.asarray(original)
  indices = np.asarray(indices)
  values = np.asarray(values)
  output = np.copy(original)
  update_indices = indices.shape[:-1]
  for idx in np.ndindex(update_indices):
    if reduction is None:
      output[tuple(indices[idx])] = values[idx]
    elif reduction == 'add':
      output[tuple(indices[idx])] += values[idx]
    elif reduction == 'mul':
      output[tuple(indices[idx])] *= values[idx]
  return output

# verify scatternd matches reference implementation.
def check_scatter_nd(params, indices, updates, reduction=None, output=None, output_shape=None, desc=None):
  y = scatter_nd(params, indices, updates, reduction=reduction)
  if output_shape is not None:
    x = np.asarray(output_shape)
    y = np.asarray(y.shape)
  elif output is not None:
    x = np.asarray(output)
  else:
    x = scatter_nd_ref(params, indices, updates, reduction=reduction)
  ntu.check_eq(x, y)
  return y


# The below Scatter's numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter_ref(data, indices, updates, axis=0, reduction=None):  # type: ignore
    if axis < 0:
        axis = data.ndim + axis

    idx_xsection_shape = indices.shape[:axis] + indices.shape[axis + 1:]

    def make_slice(arr, axis, i):  # type: ignore
        slc = [slice(None)] * arr.ndim
        slc[axis] = i
        return slc

    def unpack(packed):  # type: ignore
        unpacked = packed[0]
        for i in range(1, len(packed)):
            unpacked = unpacked, packed[i]
        return unpacked

    # We use indices and axis parameters to create idx
    # idx is in a form that can be used as a NumPy advanced indices for scattering of updates param. in data
    idx = [[unpack(np.indices(idx_xsection_shape).reshape(indices.ndim - 1, -1)),
            indices[tuple(make_slice(indices, axis, i))].reshape(1, -1)[0]] for i in range(indices.shape[axis])]
    idx = list(np.concatenate(idx, axis=1))
    idx.insert(axis, idx.pop())

    # updates_idx is a NumPy advanced indices for indexing of elements in the updates
    updates_idx = list(idx)
    updates_idx.pop(axis)
    updates_idx.insert(axis, np.repeat(np.arange(indices.shape[axis]), np.prod(idx_xsection_shape)))

    scattered = np.copy(data)
    if reduction == 'add':
      scattered[tuple(idx)] += updates[tuple(updates_idx)]
    elif reduction == 'mul':
      scattered[tuple(idx)] *= updates[tuple(updates_idx)]
    elif reduction is None:
      scattered[tuple(idx)] = updates[tuple(updates_idx)]
    else:
      raise ValueError("Unknown reduction type")
    return scattered

# verify scatter matches reference implementation.
def check_scatter(params, indices, updates, axis=0, reduction=None, output=None, output_shape=None, desc=None):
  params = np.asarray(params).astype(np.float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(np.float)
  y = scatter(params, indices, updates, axis=axis, reduction=reduction)
  if output_shape is not None:
    x = np.asarray(output_shape)
    y = np.asarray(y.shape)
  elif output is not None:
    x = np.asarray(output)
  else:
    x = scatter_ref(params, indices, updates, axis=axis, reduction=reduction)
  ntu.check_eq(x, y)
  return y

def scatter_name_func(testcase_func, param_num, param):
  kwargs = dict(param.kwargs)
  name = testcase_func.__name__ + '_' + str(param_num)
  if 'desc' in kwargs:
    desc = kwargs.pop('desc')
    name += f"  {desc!r} "
  args = ntu.get_call_args(check_scatter, *param.args, **kwargs)
  indices = np.asarray(args['indices'])
  params = np.asarray(args['params'])
  updates = np.asarray(args['updates'])
  reduction = args['reduction']
  kvs = []
  kvs.append(('params.shape', params.shape))
  kvs.append(('indices.shape', indices.shape))
  kvs.append(('updates.shape', updates.shape))
  if 'output' in kwargs:
    output = np.asarray(kwargs.pop('output'))
    kvs.append(('output.shape', output.shape))
  elif 'output_shape' in kwargs:
    output_shape = np.asarray(kwargs.pop('output_shape'))
    kvs.append(('output.shape', output_shape))
  if reduction is not None:
    kvs.append(('reduction', reduction))
  for k, v in kvs:
    name += " "
    name += str(k)
    name += "="
    name += str(v)
  return name

scatter_nd_test_case = functools.partial(ntu.parameterized.expand, name_func=scatter_name_func)
scatter_test_case = functools.partial(ntu.parameterized.expand, name_func=scatter_name_func)


class ScatterNdTestCase(ntu.TestCase):

  @scatter_nd_test_case([
    # ntu.param(
    #   desc = "Scattering individual elements",
    #   params = np.ones((8,)),
    #   indices = [[4], [3], [1], [7]],
    #   updates = [9, 10, 11, 12],
    #   # output = np.asarray([0, 11, 0, 10, 9, 0, 0, 12]) + 1,
    # ),
    *[
      ntu.param(
        desc = "Scattering individual elements",
        params = nnp.values((8,)),
        indices = [[4], [0]],
        updates = [9, 2],
        # output = np.asarray([0, 11, 0, 10, 9, 0, 0, 12]) + 1,
        reduction = reduction
      ) for reduction in [None, 'add', 'mul']
    ],
    # ntu.param(
    #   desc = "Scattering slices",
    #   params = np.ones((4, 4, 4,)),
    #   indices = [[0], [2]],
    #   updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]],
    #              [[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]]],
    #   # output =
    #   # np.asarray([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    #   #             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    #   #             [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
    #   #             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]) + 1,
    # ),
    # ntu.param(
    #   desc = "ScatterND Add",
    #   params = np.ones((4, 4, 4,)),
    #   indices = [[0], [2]],
    #   updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]],
    #              [[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]]],
    #   output = [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    #             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    #             [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    #             [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
    #   reduction = None,
    # ),
    # ntu.param(
    #   desc = "ScatterND",
    #   params = nnp.values((4, 4, 4,)),
    #   indices = [[0], [2]],
    #   updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]],
    #              [[5, 5, 5, 5], [6, 6, 6, 6],
    #               [7, 7, 7, 7], [8, 8, 8, 8]]],
    #   # output = [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    #   #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    #   #           [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
    #   #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
    #   # reduction = 'add',
    # ),
  ])
  def test_scatter_nd(self, *args, **kws):
    check_scatter_nd(*args, **kws)

class ScatterTestCase(ntu.TestCase):

  @scatter_test_case([
    *[
      ntu.param(
        desc = "ONNX ScatterElements example 1",
        params = nnp.values((3,3,)),
        indices = [
          [1, 0, 2],
          [0, 2, 1],
        ],
        updates = [
          [1.0, 1.1, 1.2],
          [2.0, 2.1, 2.2],
        ],
        # output = [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        reduction = reduction
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "ONNX ScatterElements example 2",
        params = nnp.values((1,5,)),
        indices = [[1, 3]],
        updates = [[1.1, 2.1]],
        axis = 1,
        # output = [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        reduction = reduction
      ) for reduction in [None, 'add', 'mul']
    ],
  ])
  def test_scatter(self, *args, **kws):
    check_scatter(*args, **kws)
