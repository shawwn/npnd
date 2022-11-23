# See "prefix sums and their applications" [Ble93]:
# https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf
from abc import abstractmethod

import numpy as np
import builtins as py

from . import shape as shape_lib
from . import gather_nd as gather_lib
from . import reshape as reshape_lib
from . import cumsum as cumsum_lib
from . import diff as diff_lib
from . import one_hot as one_hot_lib
from .trace import trace

import pytreez
import collections.abc as std
import dataclasses

from typing import Sequence, Tuple, Optional, Callable, NamedTuple, overload, TypeVar

T_co = TypeVar("T_co", covariant=True)

@dataclasses.dataclass
class Flattened(std.Sequence[T_co]):
    values: Tuple[T_co, ...]
    tree: pytreez.lib.PyTreeDef

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> T_co: ...

    @overload
    @abstractmethod
    def __getitem__(self, index: slice) -> Sequence[T_co]: ...

    def __getitem__(self, index: int) -> T_co:
        return self.values[index]

    def __len__(self) -> int:
        return len(self.values)

    def unflat(self):
        return self.tree.unflatten(self.values)

def flatten(vals):
    values, tree = pytreez.tree_flatten(vals)
    values = tuple(values)
    return Flattened(values, tree)

def vector(*args):
    pass


class BinOp(NamedTuple):
    def __call__(self, *args, **kwargs):
        pass

# def all_prefix_sums(binop: Callable[[Tuple[Tuple, pytreez.lib.PyTreeDef],], Tuple[Tuple, ]):
#     pass

# def nsorted(a: np.ndarray, axis=0, ascending=True, indices=True) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
#   vals = shape_lib.unstack(a, axis=axis, keepdims=KEEPDIMS)
#   idxs = ()
#   if indices:
#     idxs = shape_lib.ndshape(np.shape(a))[axis]
#     idxs = shape_lib.unstack(idxs, axis=axis, keepdims=KEEPDIMS)
#   n = np.shape(a)[axis]
#   vals, idxs, total = bitonic_sort2(vals, idxs, 0, n, direction=ascending)
#   return vals, idxs, total
