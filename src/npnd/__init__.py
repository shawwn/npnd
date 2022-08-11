# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

libs = []

def reload():
  from importlib import reload
  for lib in libs:
    reload(lib)
  import npnd
  reload(npnd)
  return npnd

import npnd.core as core
libs.append(core)

import npnd._src.config as config
libs.append(config)

import npnd._src.ops.values as values_lib
libs.append(values_lib)
from npnd._src.ops.values import (
  ones,
  ones_like,
  zeros,
  zeros_like,
  values,
  values_like,
)

import npnd._src.ops.one_hot as one_hot_lib
libs.append(one_hot_lib)
from npnd._src.ops.one_hot import one_hot

import npnd._src.ops.gather_nd as gather_nd_lib
libs.append(gather_nd_lib)
from npnd._src.ops.gather_nd import gather_elements_ref
from npnd._src.ops.gather_nd import gather_nd
from npnd._src.ops.gather_nd import gather
from npnd._src.ops.gather_nd import gather_elements
from npnd._src.ops.gather_nd import take_along_axis
from npnd._src.ops.gather_nd import take

from npnd._src.ops import meshgrid as meshgrid_lib
libs.append(meshgrid_lib)
from npnd._src.ops.meshgrid import (
  meshgrid,
  meshgrid_shapes,
)

import npnd._src.ops.scatter_nd as scatter_nd_lib
libs.append(scatter_nd_lib)
from npnd._src.ops.scatter_nd import scatter_nd
from npnd._src.ops.scatter_nd import scatter_nd_ref

import npnd._src.ops.scatter as scatter_lib
libs.append(scatter_lib)
from npnd._src.ops.scatter import scatter
from npnd._src.ops.scatter import scatter_ref

import npnd._src.ops.broadcast as broadcast_lib
libs.append(broadcast_lib)
from npnd._src.ops.broadcast import broadcast_to
from npnd._src.ops.broadcast import broadcast_in_dim
from npnd._src.ops.broadcast import broadcast_shapes
from npnd._src.ops.broadcast import broadcast_arrays

import npnd._src.ops.shape as shape_lib
libs.append(shape_lib)
from npnd._src.ops.shape import (
  shape,
  ndim,
  ndindex,
  ndindices,
  ndshape,
  flat_inner_shape,
  flat_inner_dims,
  flat_nd_indices,
  trigrid,
  tri,
  tril,
  triu,
  trilu,
  diagonal_shape,
  diag,
  diagonal,
  eye,
  identity,
  iota,
  unstack,
)


import npnd._src.ops.reshape as reshape_lib
libs.append(reshape_lib)
from npnd._src.ops.reshape import (
  reshape_shape,
  reshape,
  expand_dims,
  prod,
)

import npnd._src.ops.sort as sort_lib
libs.append(sort_lib)
from npnd._src.ops.sort import (
  argsort,
  argmin,
  argmax,
  sort,
  amin,
  amax,
  topk,
  unique,
)

import npnd._src.ops.cumsum as cumsum_lib
libs.append(cumsum_lib)
from npnd._src.ops.cumsum import (
  cumsum,
  cumprod,
)

import npnd._src.ops.diff as diff_lib
libs.append(diff_lib)
from npnd._src.ops.diff import (
  diff,
)

import npnd._src.ops.matmul as matmul_lib
libs.append(matmul_lib)
from npnd._src.ops.matmul import (
  matmul,
  matmul_shape,
)

import npnd._src.ops.mod as mod_lib
libs.append(mod_lib)
from npnd._src.ops.mod import (
  inttype,
  itrunc,
  ifloor,
  iceil,
  trunc,
  floor,
  ceil,
  fmod,
  mod,
  sign,
)

import npnd._src.ops.einsum as einsum_lib
libs.append(einsum_lib)
from npnd._src.ops.einsum import (
  einsum,
  einsum_model,
  einsum_model_test,
)
