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


def scatter_name_func(testcase_func, param_num, param):
  kwargs = dict(param.kwargs)
  name = testcase_func.__name__ + '_' + str(param_num)
  if 'desc' in kwargs:
    desc = kwargs.pop('desc')
    name += f"  {desc!r} "
  args = ntu.get_call_args(check_scatter_nd, *param.args, **kwargs)
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


class GatherNdTestCase(ntu.TestCase):

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
