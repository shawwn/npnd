def internal(*args):
  msg = "".join([x if isinstance(x, str) else repr(x) for x in args])
  raise RuntimeError(msg)

def invalid_argument(*args):
  msg = "".join([x if isinstance(x, str) else repr(x) for x in args])
  raise ValueError(msg)
