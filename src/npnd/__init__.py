# read version from installed package
from importlib.metadata import version
__version__ = version(__name__)
del version

from npnd._src.ops.values import values
from npnd._src.ops.values import values_like
from npnd._src.ops.one_hot import one_hot
from npnd._src.ops.gather_nd import gather_nd
