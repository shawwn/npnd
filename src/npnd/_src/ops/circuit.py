# "The design and verification of a sorter core"
# https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=DA89C49AA76DF3D7E6AC0A75642E27D5?doi=10.1.1.22.7187&rep=rep1&type=pdf
from __future__ import annotations

from typing import *

import numpy as np

from . import slice as slice_lib
from . import shape as shape_lib

T = TypeVar("T")

def split(l, axis=0, n=None):
    if n is None:
        n = np.shape(l)[axis]
    return np.array_split(l, n, axis=axis)

def concat(ls, axis=0):
    return np.concatenate(ls, axis=axis)

def unstack(l, axis=0):
    ls = split(l, axis=axis)
    return tuple([np.squeeze(l, axis=axis) for l in ls])

def stack(ls, axis=0):
    ls2 = [np.expand_dims(l, axis=axis) for l in ls]
    return concat(ls2, axis=axis)

def halve(l, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    return split(l, axis=axis, n=2)

def unhalve(ls: Tuple[np.ndarray, ...], axis=0):
    return concat(ls, axis=axis)

# A circuit in which g works on the b ottom half of the input list and h on the top
# half is written parl g h
def parl(bot, top):
    def network(l: np.ndarray, axis=0) -> np.ndarray:
        lh, rh = halve(l, axis=axis)
        ls = bot(lh, axis=axis), top(rh, axis=axis)
        return unhalve(ls, axis=axis)
    return network

# The special case in which the two components are the same
# arises often and so gets its own abbreviation `two`
def two(both):
    return parl(both, both)

def reverse(l, axis=0):
    # return np.flip(l, axis=axis)
    return slice_lib.slice_on_axis(l, axis, start=None, stop=None, step=-1)

def odds(l, axis=0):
    return slice_lib.slice_on_axis(l, axis, start=0, stop=None, step=2)

def evens(l, axis=0):
    return slice_lib.slice_on_axis(l, axis, start=1, stop=None, step=2)

def at(l, i):
    if i < 0:
        i += len(l)
    if i >= 0 and i < len(l):
        return l[i]

def riffle(ls: Tuple[np.ndarray, ...], axis=0) -> np.ndarray:
    parts = [split(l, axis=axis) for l in ls]
    outs = []
    n = max([len(part) for part in parts])
    for j in range(n):
        for i in range(len(parts)):
            p = parts[i]
            if j < len(p):
                outs.append(p[j])
    return concat(outs, axis=axis)

def unriffle(l: np.ndarray, axis=0) -> Tuple[np.ndarray, np.ndarray]:
    lh = odds(l, axis=axis)
    rh = evens(l, axis=axis)
    ls = rh, lh
    # ls = lh, rh
    # return unhalve(ls, axis=axis)
    return ls

# Related to two, we also introduce the pattern ilv for interleave.
# Whereas two f applies f to the top and bottom halves of a list,
# ilv f applies f to the odd and even elements.

def interleave(odd, even):
    def network(l, axis=0):
        lh = odds(l, axis=axis)
        rh = evens(l, axis=axis)
        if np.shape(lh)[axis] == np.shape(rh)[axis]:
            ls = even(rh, axis=axis), odd(lh, axis=axis)
        else:
            ls = odd(lh, axis=axis), even(rh, axis=axis)
        # return unhalve(ls, axis=axis)
        return riffle(ls, axis=axis)
    return network

def identity(l, axis=0):
    return l

def idfunc(f):
    def network(l: np.ndarray, axis=0):
        if isinstance(l, (tuple, list)):
            return type(l)([f(v, axis=axis) for v in l])
        else:
            return f(l, axis=axis)
    return network

def pipeline(*fs):
    def network(l: np.ndarray, axis=0) -> np.ndarray:
        for f in fs:
            l = f(l, axis=axis)
        return l
    return network

# def ilv(f):
#     return pipeline(unriffle, idfunc(f), riffle)

def ilv(f):
    return interleave(f, f)


# Now we are in a position to define a connection
# pattern for butterfly circuits
# bfly 1 f = f
# bfly n f = ilv (bfly (n - 1) f) >-> evens f
def bfly(n, f):
    if n <= 1:
        return f
    raise NotImplementedError

def twosort(ls: Tuple[np.ndarray, np.ndarray], axis=0):
    assert len(ls) == 2
    assert axis == 0
    a, b = ls
    c = np.less_equal(a, b)
    x = np.where(c, a, b)
    y = np.where(np.bitwise_not(c), a, b)
    return x, y

def nsort2(vals: List[np.ndarray], idxs: List[np.ndarray], ascending: bool = True):
    assert len(vals) == 2
    assert len(idxs) == 2
    assert isinstance(vals, list)
    assert isinstance(idxs, list)
    valA, valB = vals
    idxA, idxB = idxs
    mask = (np.less_equal if ascending else np.greater_equal)(valA, valB)
    flip = np.bitwise_not(mask)
    val_min = np.where(mask, valA, valB)
    val_max = np.where(flip, valA, valB)
    idx_min = np.where(mask, idxA, idxB)
    idx_max = np.where(flip, idxA, idxB)
    vals[0] = val_min
    vals[1] = val_max
    idxs[0] = idx_min
    idxs[1] = idx_max
    # val = val_min, val_max
    # idx = idx_min, idx_max
    # return val, idx


def nsort(vals: Tuple[np.ndarray, ...], idxs: Tuple[np.ndarray, ...], ascending: bool = True) -> Tuple[Tuple[np.ndarray, ...], Tuple[np.ndarray, ...], int]:
    if len(vals) <= 1:
        return vals, idxs, 0
    elif len(vals) == 2:
        assert len(idxs) == 2
        valA, valB = vals
        idxA, idxB = idxs
        if ascending:
            mask = np.less_equal(valA, valB)
        else:
            mask = np.greater_equal(valA, valB)
        flip = np.bitwise_not(mask)
        val_min = np.where(mask, valA, valB)
        idx_min = np.where(mask, idxA, idxB)
        val_max = np.where(flip, valA, valB)
        idx_max = np.where(flip, idxA, idxB)
        val = val_min, val_max
        idx = idx_min, idx_max
        return val, idx, 1
    else:
        [*valBs, valC] = vals
        [*idxBs, idxC] = idxs
        total = 0
        for i in range(len(valBs)):
            if False:
                valB = valBs[i]
                idxB = idxBs[i]
                (valB, valC), (idxB, idxC), n = nsort((valB, valC), (idxB, idxC))
                total += n
                valBs[i] = valB
                idxBs[i] = idxB
            elif True:
                valBC = valBs[i], valC
                idxBC = idxBs[i], idxC
                valBC, idxBC, n = nsort(valBC, idxBC, ascending=ascending)
                total += n
                valBs[i] = valBC[0]
                idxBs[i] = idxBC[0]
                valC = valBC[1]
                idxC = idxBC[1]
            else:
                valBC = [valBs[i], valC]
                idxBC = [idxBs[i], idxC]
                nsort2(valBC, idxBC, ascending=ascending)
                total += 1
                valBs[i] = valBC[0]
                idxBs[i] = idxBC[0]
                valC = valBC[1]
                idxC = idxBC[1]
        valCs, idxCs, m = nsort(valBs, idxBs, ascending=ascending)
        total += m
        valCs += (valC,)
        idxCs += (idxC,)
        return valCs, idxCs, total

def sorted(vals: np.ndarray, axis=0, ascending=True) -> Tuple[np.ndarray, np.ndarray, int]:
    idxs = shape_lib.ndshape(np.shape(vals))[axis]
    idxs = unstack(idxs, axis=axis)
    vals = unstack(vals, axis=axis)
    vals, idxs, comparisons = nsort(vals, idxs, ascending=ascending)
    vals = stack(vals, axis=axis)
    idxs = stack(idxs, axis=axis)
    return vals, idxs, comparisons

def threesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray], axis=0):
    x0, y0, z0 = ls
    x1, y1 = twosort((x0, y0), axis=axis)
    y2, z1 = twosort((y1, z0), axis=axis)
    x2, y3 = twosort((x1, y2), axis=axis)
    return x2, y3, z1

def threesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray], axis=0):
    x0, y0, z0 = ls
    x1, y1 = twosort((x0, y0), axis=axis)
    x2, z1 = twosort((x1, z0), axis=axis)
    y2, z2 = twosort((y1, z1), axis=axis)
    return x2, y2, z2

# def threesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray], axis=0):
#     x0, y0, z0 = ls
#     x1, y1 = twosort((x0, y0), axis=axis)
#     x2, z1 = twosort((x1, z0), axis=axis)
#     x3, y2 = twosort((x2, y1), axis=axis)
#     return x3, y2, z1

def foursort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
    a0, b0, c0, d0 = ls

    a1, b1 = twosort((a0, b0), axis=axis)
    c1, d1 = twosort((c0, d0), axis=axis)

    a2, c2 = twosort((a1, c1), axis=axis)
    b2, d2 = twosort((b1, d1), axis=axis)

    a3, b3 = twosort((a2, b2), axis=axis)
    c3, d3 = twosort((c2, d2), axis=axis)

    b4, c4 = twosort((b3, c3), axis=axis)

    return a3, b4, c4, d3

# def foursort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
#     a0, b0, c0, d0 = ls
#     a2, b2, d2 = threesort((a0, b0, d0), axis=axis)
#     b4, c4, d2 = threesort((b2, c2, d0), axis=axis)
#     a3, d3 = twosort((a2, d2), axis=axis)
#     return a3, b4, c4, d3

# def fivesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
#     a, b, c, d, e = ls
#     a2, b2, e2 = threesort((a, b, e), axis=axis)
#     c2, d2, e4 = threesort((c, d, e2), axis=axis)
#     a5, b6, c6, d5 = foursort((a2, b2, c2, d2), axis=axis)
#     return a5, b6, c6, d5, e4

def sixsort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
    a, b, c, d, e, f = ls

    e2, f1 = twosort((e, f), axis=axis)
    d2, f2 = twosort((d, f1), axis=axis)
    c2, f3 = twosort((c, f2), axis=axis)
    b2, f4 = twosort((b, f3), axis=axis)
    a2, f5 = twosort((a, f4), axis=axis)
    a7, b8, c8, d7, e6 = fivesort((a2, b2, c2, d2, e2), axis=axis)
    return a7, b8, c8, d7, e6, f5

    # a, b, e, f = foursort((a, b, e, f), axis=axis)
    # c, d, e, f = foursort((c, d, e, f), axis=axis)
    # a5, b6, c6, d5 = foursort((a2, b2, c2, d2), axis=axis)
    # return a5, b6, c6, d5, e4

def fivesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
    a, b, c, d, e = ls

    d2, e1 = twosort((d, e), axis=axis)
    c2, e2 = twosort((c, e1), axis=axis)
    b2, e3 = twosort((b, e2), axis=axis)
    a2, e4 = twosort((a, e3), axis=axis)

    a5, b6, c6, d5 = foursort((a2, b2, c2, d2), axis=axis)
    return a5, b6, c6, d5, e4
    #
    # b, c = twosort((b, c), axis=axis)
    # a, b = twosort((a, b), axis=axis)
    #
    # b, c = twosort((b, c), axis=axis)
    # a, b = twosort((a, b), axis=axis)
    #
    # a, b = twosort((a, b), axis=axis)

    # b, c, d = threesort((b, c, d), axis=axis)
    # a2, b2, e2 = threesort((a, b, e), axis=axis)
    # c2, d2, e4 = threesort((c, d, e2), axis=axis)
    # c3, d3 = twosort((c2, d2), axis=axis)
    # a5, b6, c6, d5 = foursort((a2, b2, c2, d2), axis=axis)
    # return a5, b6, c6, d5, e4
    # return a, b, c, d, e

# def fivesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
#     a0, b0, c0, d0, e0 = ls
#
#     a2, b3,         e1 = threesort((a0, b0, e0), axis=axis)
#     if True:
#             c2, d3, e2 = threesort((c0, d0, e1), axis=axis)
#     a5, b7, c6, d6 = foursort((a2, b3, c2, d3), axis=axis)
#     return a5, b7, c6, d6, e2
#     c3, d4 = twosort((c2, d3), axis=axis)
#     a4, b6, c4 = threesort((a2, b3, c3), axis=axis)
#     return a4, b6, c4, d4, e2
#
#
#     # a2, c3, e1 = threesort((a0, c0, e0), axis=axis)
#     # # a2, b0, c3, d0, e1
#     # a4, b3, c4 = threesort((a2, b0, c3), axis=axis)
#     # # a4, b3, c4, d0, e1
#     # a6, b6, d1 = threesort((a4, b3, d0), axis=axis)
#     # # a6, b6, c4, d1, e1
#     #
#     # a2, b3, c1 = threesort((a0, b0, c0), axis=axis)
#     # b7, c5, d4, e3 = foursort((
#     #
#     # a1, b1 = twosort((a0, b0), axis=axis)
#     # d1, e1 = twosort((d0, e0), axis=axis)
#     #
#     # b3, c2, e2 = threesort((b1, c1, d1), axis=axis)
#     #
#     # a2, c2 = twosort((a1, c1), axis=axis)
#     # b2, d2 = twosort((b1, d1), axis=axis)
#     #
#     # a3, b3 = twosort((a2, b2), axis=axis)
#     # c3, d3 = twosort((c2, d2), axis=axis)
#     #
#     # b4, c4 = twosort((b3, c3), axis=axis)
#     #
#     # return a3, b4, c4, d3

# def fivesort(ls: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], axis=0):
#     a0, b0, c0, d0, e0 = ls
#
#     a2, b3, c1 = threesort((a0, b0, c0), axis=axis)
#     c2, d3, e1 = threesort((c1, d0, e0), axis=axis)
#
#     # return a2, b3, c2, d3, e1
#     a4, b5, c3 = threesort((a2, b3, c2), axis=axis)
#     # return a4, b5, c3, d3, e1
#     c4, d5, e2 = threesort((c3, d3, e1), axis=axis)
#     return a4, b5, c4, d5, e2
#
#     # if True:
#     #     b2, c3, d1 = threesort((b0, c0, d0), axis=axis)
#     #
#     # a1, b2, c2 = threesort((a0, b2, c3), axis=axis)
#     # c3, d2, e1 = threesort((c2, d1, e0), axis=axis)
#     #
#     # return a1, b2, c3, d2, e1
