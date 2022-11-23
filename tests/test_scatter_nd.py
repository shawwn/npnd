import numpy as np

import npnd
import npnd.test_util as ntu
import npnd.numpy as nnp
import functools
import inspect
try:
  import torch
except ImportError:
  torch = None

# verify scatternd matches reference implementation.
def check_scatter_nd(params, indices, updates, reduction=None, output=None, output_shape=None, desc=None, debug=False):
  params = np.asarray(params).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(float)
  if debug:
    breakpoint()
  x = npnd.scatter_nd_ref(params, indices, updates, reduction=reduction)
  y = npnd.scatter_nd(params, indices, updates, reduction=reduction)
  if output_shape is not None:
    x = np.asarray(output_shape)
    y = np.asarray(y.shape)
  elif output is not None:
    x = np.asarray(output)
  ntu.check_eq(x, y)
  return y

# verify scatter matches reference implementation.
def check_scatter(params, indices, updates, axis=0, reduction=None, output=None, output_shape=None, desc=None, debug=False):
  params = np.asarray(params).astype(float)
  indices = np.asarray(indices).astype(np.int64)
  updates = np.asarray(updates).astype(float)
  if debug:
    breakpoint()
  x = npnd.scatter_ref(params, indices, updates, axis=axis, reduction=reduction)
  y = npnd.scatter(params, indices, updates, axis=axis, reduction=reduction)
  if output_shape is not None:
    x = np.asarray(output_shape)
    y = np.asarray(y.shape)
  elif output is not None:
    x = np.asarray(output)
  ntu.check_eq(x, y)
  return y


# The below Scatter's numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter_ref(data, indices, updates, axis=0, reduction=None):  # type: ignore
    if torch is not None:
      options = dict(reduce=reduction) if reduction is not None else dict()
      if options.get('reduce') == 'mul':
        options['reduce'] = 'multiply'
      data = torch.tensor(data).float()
      indices = torch.tensor(indices).long()
      updates = torch.tensor(updates).float()
      return data.scatter(axis, indices, updates, **options)
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

    def make_indices_for_duplicate(idx):  # type: ignore
        final_idx = list()
        for i in range(len(idx[0])):
            final_idx.append(tuple(idx_element[i] for idx_element in idx))
        return list(final_idx)

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
    if reduction is None:
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    else:
        idx, updates_idx = make_indices_for_duplicate(idx), make_indices_for_duplicate(updates_idx)
        for iter, idx_set in enumerate(idx):
            if reduction == 'add':
                scattered[idx_set] += updates[updates_idx[iter]]
            elif reduction == 'mul':
                scattered[idx_set] *= updates[updates_idx[iter]]
            else:
                raise ValueError("Unknown reduction type")
    return scattered

# The below ScatterElements' numpy implementation is from https://stackoverflow.com/a/46204790/11767360
def scatter_elements_ref(data, indices, updates, axis=0, reduction='none'):  # type: ignore
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

    def make_indices_for_duplicate(idx):  # type: ignore
        final_idx = list()
        for i in range(len(idx[0])):
            final_idx.append(tuple(idx_element[i] for idx_element in idx))
        return list(final_idx)

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
    if reduction == 'none':
        scattered[tuple(idx)] = updates[tuple(updates_idx)]
    else:
        idx, updates_idx = make_indices_for_duplicate(idx), make_indices_for_duplicate(updates_idx)
        for iter, idx_set in enumerate(idx):
            if reduction == 'add':
                scattered[idx_set] += updates[updates_idx[iter]]
            elif reduction == 'mul':
                scattered[idx_set] *= updates[updates_idx[iter]]
    return scattered


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
    ntu.param(
      desc = "Scattering slices",
      params = np.ones((4, 4, 4,)),
      indices = [[0], [2]],
      updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
                  [7, 7, 7, 7], [8, 8, 8, 8]],
                 [[5, 5, 5, 5], [6, 6, 6, 6],
                  [7, 7, 7, 7], [8, 8, 8, 8]]],
      # output =
      # np.asarray([[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
      #             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
      #             [[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
      #             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]) + 1,
    ),
    *[
      ntu.param(
        desc = "ScatterND slice reductions",
        params = np.ones((4, 4, 4,)),
        indices = [[0], [2]],
        updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
                    [7, 7, 7, 7], [8, 8, 8, 8]],
                   [[5, 5, 5, 5], [6, 6, 6, 6],
                    [7, 7, 7, 7], [8, 8, 8, 8]]],
        # output = [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
        #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        #           [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
        #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "ScatterND slice reductions #2",
        params = nnp.values((4, 4, 4,)),
        indices = [[0], [2]],
        updates = [[[5, 5, 5, 5], [6, 6, 6, 6],
                    [7, 7, 7, 7], [8, 8, 8, 8]],
                   [[5, 5, 5, 5], [6, 6, 6, 6],
                    [7, 7, 7, 7], [8, 8, 8, 8]]],
        # output = [[[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
        #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        #           [[6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8], [9, 9, 9, 9]],
        #           [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "tensor_scatter_nd_update scalar updates",
        params = nnp.values_like([0, 0, 0, 0, 0, 0, 0, 0]),
        indices = [[1], [3], [4], [7]],
        updates = [9, 10, 11, 12],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "tensor_scatter_nd_update scalar updates #2",
        params = nnp.values_like([[1, 1], [1, 1], [1, 1]]),
        indices = [[0, 1], [2, 0]],
        updates = [5, 10],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "tensor_scatter_nd_update slice updates #1",
        params = nnp.values((6, 3)),
        indices = [[2], [4]],
        updates = [[1, 2, 3], [4, 5, 6]],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "tensor_scatter_nd_update More slice update examples #1",
        params = nnp.values((13,11,7,5,3)),
        indices = [[0],[1]],
        # updates = nnp.values((2, 11,7,5,3)),
        updates = nnp.ones((2, 11,7,5,3)),
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul']
    ],
    *[
      ntu.param(
        desc = "tensor_scatter_nd_update More slice update examples (drawing an X)",
        params = nnp.values((5,5)),
        indices = [
          [[0,0],
           [1,1],
           [2,2],
           [3,3],
           [4,4]],
          [[0,4],
           [1,3],
           [2,2],
           [3,1],
           [4,0]],
          [[3,0],
           [1,0],
           [2,0],
           [0,0],
           [4,0]],
        ],
        updates = nnp.values_like([
          [1,1,1,1,1],
          [1,1,1,1,1],
          [1,1,1,1,1],
        ]) + 1,
        reduction = reduction,
        # debug=True,
      ) for reduction in [None, 'add']
      # ) for reduction in [None, 'add', 'mul']
    ],
  ])
  def test_scatter_nd(self, *args, **kws):
    check_scatter_nd(*args, **kws)

  # https://www.tensorflow.org/api_docs/python/tf/tensor_scatter_nd_update#more_slice_update_examples_2
  def test_scatter_nd_2(self):
    batch_size, time, width, height, channels = 13,11,7,5,3
    video_batch = nnp.values([batch_size, time, width, height, channels])
    indices = [[0],[1]]
    # new_clips = nnp.values([2, time, width, height, channels])
    new_clips = nnp.ones([2, time, width, height, channels])
    for reduction in [None, 'add', 'mul']:
      check_scatter_nd(video_batch, indices, new_clips, reduction=reduction)
    indices = [[0, 0], [1, 0], [2, 0]] # num_updates=3, index_depth=2
    new_images = np.ones([
      # num_updates=3, inner_shape=(width, height, channels)
      3, width, height, channels])
    for reduction in [None, 'add', 'mul']:
      check_scatter_nd(video_batch, indices, new_images, reduction=reduction)

  def test_scatter_nd_3(self):
    tensor = nnp.values((5,5))
    indices = [
      [[0,0],
       [1,1],
       [2,2],
       [3,3],
       [4,4]],
      [[0,4],
       [1,3],
       [2,2],
       [3,1],
       [4,0]],
      [[3,0],
       [1,0],
       [2,0],
       [0,0],
       [4,0]],
      [[0,3],
       [0,1],
       [0,2],
       [0,0],
       [0,4]],
    ]
    updates = nnp.values_like([
      [1,1,1,1,1],
      [1,1,1,1,1],
      [1,1,1,1,1],
      [1,1,1,1,1],
    ]) + 1
    # for reduction in [None, 'add']:#, 'mul']:
    for reduction in [None]:#, 'mul']:
      check_scatter_nd(tensor, indices, updates, reduction=reduction)



class ScatterTestCase(ntu.TestCase):

  @scatter_test_case([
    *[
      ntu.param(
        desc = "ONNX ScatterElements",
        # params = nnp.values((3,3,)),
        params = np.ones((3,3,)),
        indices = [
          [1, 0, 2],
          # [0, 2, 1],
        ],
        updates = [
          [3.0, 0.0, 3.2],
          # [2.0, 2.1, 2.2],
        ],
        # output = [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        reduction = reduction
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    *[
      ntu.param(
        desc = "ONNX ScatterElements",
        params = nnp.values((3,3,)) + 1,
        # params = np.ones((3,3,)),
        indices = [
          [1, 0, 2],
          # [0, 2, 1],
        ],
        updates = [
          [3.0, 0.0, 3.2],
          # [2.0, 2.1, 2.2],
        ],
        # output = [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        axis=axis,
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul', 'min', 'max'] for axis in [0, 1]
    ],
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
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    # *[
    #   ntu.param(
    #     desc = "ONNX ScatterElements example 2",
    #     params = nnp.values((1,5,)),
    #     indices = [[1, 3], [2, 4]],
    #     updates = [[1.1, 2.1], [5.1, 6.1]],
    #     axis = 1,
    #     # output = [[1.0, 1.1, 3.0, 2.1, 5.0]],
    #     reduction = reduction
    #   ) for reduction in [None, 'add', 'mul']
    # ],
    *[
      ntu.param(
        desc = "ONNX scatter_elements_with_axis",
        params = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        indices = np.array([[1, 3]], dtype=np.int64),
        updates = np.array([[1.1, 2.1]], dtype=np.float32),
        axis = 1,
        # output = [[1.0, 1.1, 3.0, 2.1, 5.0]],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    *[
      ntu.param(
        desc = "ONNX scatter_elements_with_duplicate_indices",
        params = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        indices = np.array([[1, 1]], dtype=np.int64),
        updates = np.array([[1.1, 2.1]], dtype=np.float32),
        axis = 1,
        # output = [[1., 5.2, 3., 4., 5.]] if reduction == 'add' else [[1., 6.4, 3., 4., 5.]],
        reduction = reduction,
        # debug=True,
      # ) for reduction in ['add', 'mul']
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    *[
      ntu.param(
        desc = "ONNX scatter_elements_with_negative_indices",
        params = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        indices = np.array([[1, -3]], dtype=np.int64),
        updates = np.array([[1.1, 2.1]], dtype=np.float32),
        axis = 1,
        # output =
        # [[1.0, 1.1, 2.1, 4.0, 5.0]] if reduction is None else
        # [[1.0, 3.1, 5.1, 4.0, 5.0]] if reduction == 'add' else
        # [[1.0, 2.2, 6.3, 4.0, 5.0]],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    *[
      ntu.param(
        desc = "ONNX scatter_elements_without_axis",
        params = nnp.values((3,3)) + 1,
        indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64),
        updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32),
        # output =
        # [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        reduction = reduction,
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
    # *[
    #   ntu.param(
    #     desc = "Torch scatter example 1",
    #     params = nnp.values((2,5,)),
    #     indices = [[0, 1, 2, 0]],
    #     updates = nnp.values((2,5)),
    #     axis = 0,
    #     # output = [
    #     #   [2.0, 1.1, 0.0],
    #     #   [1.0, 0.0, 2.2],
    #     #   [0.0, 2.1, 1.2],
    #     # ],
    #     reduction = reduction
    #   ) for reduction in [None, 'add', 'mul']
    # ],
    # *[
    #   ntu.param(
    #     desc = "Tensorflow ScatterUpdate unit test 1",
    #     params = nnp.values((5,3,)),
    #     indices = [0, 4, 2],
    #     updates = [[100, 101, 102],
    #                [777, 778, 779],
    #                [10000, 10001, 10002]],
    #     axis = 0,
    #     # output = [
    #     #   [2.0, 1.1, 0.0],
    #     #   [1.0, 0.0, 2.2],
    #     #   [0.0, 2.1, 1.2],
    #     # ],
    #     reduction = reduction
    #   ) for reduction in [None, 'add', 'mul']
    # ],
    *[
      ntu.param(
        desc = "ONNX ScatterElements example 1 with zeros",
        params = nnp.values((3,3,)) + 1,
        indices = [
          [1, 0, 2],
          [0, 2, 1],
        ],
        updates = [
          [1.0, 0.0, 1.2],
          [2.0, 2.1, 2.2],
        ],
        # output = [
        #   [2.0, 1.1, 0.0],
        #   [1.0, 0.0, 2.2],
        #   [0.0, 2.1, 1.2],
        # ],
        reduction = reduction
      ) for reduction in [None, 'add', 'mul', 'min', 'max']
    ],
  ])
  def test_scatter(self, *args, **kws):
    check_scatter(*args, **kws)

if __name__ == '__main__':
  ntu.main()
