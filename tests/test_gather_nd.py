import numpy as np

from npnd import gather_nd_lib
import npnd.test_util as ntu
import npnd.numpy as nnp
import functools

# verify gathernd matches reference implementation.
def check_gather_nd(params, indices, batch_dims=0, output=None, output_shape=None, desc=None):
  y = gather_nd_lib.gather_nd(params, indices, batch_dims=batch_dims)
  if output_shape is not None:
    x = np.asarray(output_shape)
    y = np.asarray(y.shape)
  elif output is not None:
    x = np.asarray(output)
  else:
    x = gather_nd_lib.gather_nd_ref(params, indices, batch_dims=batch_dims)
  ntu.check_eq(x, y)
  return y

def gather_nd_name_func(testcase_func, param_num, param):
  kwargs = dict(param.kwargs)
  name = testcase_func.__name__ + '_' + str(param_num)
  if 'desc' in kwargs:
    desc = kwargs.pop('desc')
    name += f"  {desc!r} "
  args = ntu.get_call_args(check_gather_nd, *param.args, **kwargs)
  indices = np.asarray(args['indices'])
  params = np.asarray(args['params'])
  batch_dims = args['batch_dims']
  kvs = []
  kvs.append(('params.shape', params.shape))
  kvs.append(('indices.shape', indices.shape))
  if batch_dims != 0:
    kvs.append(('batch_dims', batch_dims))
  for k, v in kvs:
    name += " "
    #name += ntu.parameterized.to_safe_name(str(k)).strip('_')
    name += str(k)
    name += "="
    #name += ntu.parameterized.to_safe_name(str(v)).strip('_').replace('_', 'x')
    name += str(v)
  return name
  # return "%s_%s" %(
  #   testcase_func.__name__,
  #   ntu.parameterized.to_safe_name("_".join(str(x) for x in param.args)),
  # )

gather_nd_test_case = functools.partial(ntu.parameterized.expand, name_func=gather_nd_name_func)

# test data.
data1    = [1, 2, 3, 4, 5, 6, 7, 8]
indices1 = [[4], [3], [1], [7]]
updates1 = [9, 10, 11, 12]
output1  = [1, 11, 3, 10, 9, 6, 7, 12]
data1, indices1, updates1 = [np.asarray(x) for x in (data1, indices1, updates1)]

data2    = [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
            [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]]
indices2 = [[0], [2]]
updates2 = [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
            [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]]
data2, indices2, updates2 = [np.asarray(x) for x in (data2, indices2, updates2)]

data3    = np.stack([data1,data1]).T
indices3 = indices1.copy()
updates3 = np.stack([updates1,updates1]).T
data3, indices3, updates3 = [np.asarray(x) for x in (data3, indices3, updates3)]

data4 = -1 - nnp.values_like(data3)
indices4 = indices3.copy()
updates4 = nnp.values_like(updates3)


class GatherNdTestCase(ntu.TestCase):
  @gather_nd_test_case([
    (data1, indices1),
    (data2, indices2),
    (data3, indices3),
    (data4, indices4),
  ])
  def test_gather_nd(self, *args, **kws):
    check_gather_nd(*args, **kws)

  # ONNX GatherND from https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
  @gather_nd_test_case([
    # Example 1
    ntu.param(
      batch_dims = 0,
      params  = [[0,1],[2,3]],   # params_shape = [2, 2]
      indices = [[0,0],[1,1]],   # indices_shape = [2, 2]
      output  = [0,3]            # output_shape = [2]
    ),
    # Example 2
    ntu.param(
      batch_dims = 0,
      params  = [[0,1],[2,3]],   # params_shape = [2, 2]
      indices = [[1],[0]],       # indices_shape = [2, 2]
      output  = [[2,3],[0,1]],   # output_shape = [2]
    ),
    # Example 3
    ntu.param(
      batch_dims = 0,
      params  = [[[0,1],[2,3]],[[4,5],[6,7]]],  # params_shape = [2, 2, 2]
      indices = [[0,1],[1,0]],   # indices_shape = [2, 2]
      output  = [[2,3],[4,5]],   # output_shape = [2, 2]
    ),
    # Example 4
    ntu.param(
      batch_dims = 0,
      params  = [[[0,1],[2,3]],[[4,5],[6,7]]],  # params_shape = [2, 2, 2]
      indices = [[[0,1],[1,0]]],   # indices_shape = [2, 1, 2]
      output  = [[[2,3],[4,5]]],   # output_shape = [2, 1, 2]
    ),
    # Example 5
    ntu.param(
      batch_dims = 1,
      params  = [[[0,1],[2,3]],[[4,5],[6,7]]],  # params_shape = [2, 2, 2]
      indices = [[1],[0]],       # indices_shape = [2, 1]
      output  = [[2,3],[4,5]],   # output_shape = [2, 2]
    ),
  ], name_func=gather_nd_name_func)
  def test_ONNX_GatherND_examples(self, *args, **kws):
    check_gather_nd(*args, **kws)

  # tf.gather_nd examples from https://www.tensorflow.org/api_docs/python/tf/gather_nd

  # Gathering scalars
  @gather_nd_test_case([
    ntu.param(
      indices=[[0, 0],
               [1, 1]],
      params = [['a', 'b'],
                ['c', 'd']],
      output = ['a', 'd'],
    ),
  ])
  def test_tf_gather_nd_scalars(self, *args, **kws):
    check_gather_nd(*args, **kws)

  # Gathering slices
  @gather_nd_test_case([
    ntu.param(
      indices = [[1],
                 [0]],
      params = [['a', 'b', 'c'],
                ['d', 'e', 'f']],
      output = [['d', 'e', 'f'],
                ['a', 'b', 'c']],
    ),
  ])
  def test_tf_gather_nd_slices(self, *args, **kws):
    check_gather_nd(*args, **kws)

  # Batches
  @gather_nd_test_case([
    ntu.param(
      indices = [[0, 1],
                 [1, 0],
                 [2, 4],
                 [3, 2],
                 [4, 1]],
      params = nnp.values((5,7,3)),
      output= [[ 3,  4,  5],
               [21, 22, 23],
               [54, 55, 56],
               [69, 70, 71],
               [87, 88, 89]],
    ),
    ntu.param(
      batch_dims=1,
      indices = [[1],
                 [0],
                 [4],
                 [2],
                 [1]],
      params = nnp.values((5,7,3)),
      output= [[ 3,  4,  5],
               [21, 22, 23],
               [54, 55, 56],
               [69, 70, 71],
               [87, 88, 89]],
    ),
  ])
  def test_tf_gather_nd_batches(self, *args, **kws):
    check_gather_nd(*args, **kws)

  # More examples
  @gather_nd_test_case([
    ntu.param(
      desc = "Indexing into a 3-tensor",
      indices = [[1]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [[['a1', 'b1'],
                 ['c1', 'd1']]],
    ),
    ntu.param(
      desc = "Indexing into a 3-tensor",
      indices = [[0, 1], [1, 0]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [['c0', 'd0'],
                ['a1', 'b1']],
    ),
    ntu.param(
      desc = "Indexing into a 3-tensor",
      indices = [[0, 0, 1], [1, 0, 1]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = ['b0', 'b1'],
    ),
    ntu.param(
      desc = "Batched indexing into a matrix",
      indices = [[[0, 0]], [[0, 1]]],
      params = [['a', 'b'], ['c', 'd']],
      output = [['a'],
                ['b']],
    ),
    ntu.param(
      desc = "Batched slice indexing into a matrix",
      indices = [[[1]], [[0]]],
      params = [['a', 'b'], ['c', 'd']],
      output = [[['c', 'd']],
                [['a', 'b']]],
    ),
    ntu.param(
      desc = "Batched indexing into a 3-tensor",
      indices = [[[1]], [[0]]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [[[['a1', 'b1'],
                  ['c1', 'd1']]],
                [[['a0', 'b0'],
                  ['c0', 'd0']]]],
    ),
    ntu.param(
      desc = "Batched indexing into a 3-tensor",
      indices = [[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [[['c0', 'd0'],
                 ['a1', 'b1']],
                [['a0', 'b0'],
                 ['c1', 'd1']]],
    ),
    ntu.param(
      desc = "Batched indexing into a 3-tensor",
      indices = [[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [['b0', 'b1'],
                ['d0', 'c1']],
    ),
    ntu.param(
      desc = "Examples with batched 'params' and 'indices'",
      batch_dims = 1,
      indices = [[1],
                 [0]],
      params = [[['a0', 'b0'],
                 ['c0', 'd0']],
                [['a1', 'b1'],
                 ['c1', 'd1']]],
      output = [['c0', 'd0'],
                ['a1', 'b1']],
    ),
    ntu.param(
      desc = "Examples with batched 'params' and 'indices'",
      batch_dims = 1,
      indices = [[[1]], [[0]]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [[['c0', 'd0']],
                [['a1', 'b1']]],
    ),
    ntu.param(
      desc = "Examples with batched 'params' and 'indices'",
      batch_dims = 1,
      indices = [[[1, 0]], [[0, 1]]],
      params = [[['a0', 'b0'], ['c0', 'd0']],
                [['a1', 'b1'], ['c1', 'd1']]],
      output = [['c0'],
                ['b1']],
    ),
  ])
  def test_tf_gather_nd_more_examples(self, *args, **kws):
    check_gather_nd(*args, **kws)


if __name__ == '__main__':
  ntu.main()
