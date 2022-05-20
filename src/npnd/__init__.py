# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from numpy import (
  zeros as zeros,
  zeros_like as zeros_like,
  ones as ones,
  ones_like as ones_like,
)

from npnd.ops.gather_nd import gather_nd
from npnd.ops.one_hot import one_hot
from npnd.ops.values import (
  values as values,
  values_like as values_like,
)
