from typing import *

import contextvars as CV
import contextlib

T = TypeVar("T")

@contextlib.contextmanager
def CV_let(var: CV.ContextVar[T], val: T):
    prev = var.set(val)
    try:
        yield
    finally:
        var.reset(prev)

indentation_v = globals().setdefault("indentation_v", CV.ContextVar("indentation_v", default=""))

def indentation():
    return indentation_v.get()

def with_indent(by="  "):
    return CV_let(indentation_v, indentation() + by)

def trace(x, *args):
    ok = x is not None
    if not ok:
        return contextlib.nullcontext()
    if ok:
        margin = indentation()
        s = str(x)
        s = '\n'.join([margin + line for line in s.splitlines()])
        print(s, *args)
    ind = "  "
    # ind = f"{len(indentation())//3}> "
    return with_indent(ind if ok else '')


