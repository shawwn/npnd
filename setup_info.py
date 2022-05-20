# -*- coding: utf-8 -*-
import os
import re
import ast
import configparser
import json

def path(to):
  return os.path.join(os.path.dirname(__file__), to)

# https://gist.github.com/bpeterso2000/11277541
QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")

def strip_inline_comments(s):
  if m := re.search(QUOTED_STRING_RE, s):
    return s[:m.end()] + strip_inline_comments(s[m.end():])
  return re.sub(r'(.+?)#.*$', r'\1', s)

def ev(x):
  x = strip_inline_comments(x)
  x = x.strip()
  return False if x == 'false' else True if x == 'true' else ast.literal_eval(x)

def toml_to_config(s):
  # convert toml dict to python dict
  s = re.sub(re.compile('(?<={)(?:\s*(.+?)\s*=\s*(.+?)\s*[,])*(?:\s*(.+?)\s*=\s*(.+?)\s*)(?=})'), lambda m: (json.dumps(m.group(1)) + ": " + m.group(2)), s)
  s = s.replace('\n]', ']').replace('[\n', '[').replace(',\n', ', ')
  config = configparser.ConfigParser()
  config.read_string(s)
  config = {k.lower(): {k1.lower(): ev(v1) for k1, v1 in v.items()} for k, v in config.items()}
  return config

config = toml_to_config(open(path('pyproject.toml')).read())
info = config['tool.poetry']
package = info['name'].replace('-', '_')
author, author_email = re.findall(r"^(.+?)\s*(?:[<](.+?)[>])?$", info['authors'][0])[0]



base_kwargs = {
    'name': info['name'],
    'version': info['version'],
    'description': info.get('description'),
    'long_description': open(path(info.get('readme', 'README.md'))).read(),
    'author': author,
    'author_email': author_email,
    'url': info.get('homepage'),
    'license': info.get('license'),
}

# prune blank strings or None values
base_kwargs = {k: v for k, v in base_kwargs.items() if v}

# defaults

package_dir = \
{'': 'src'}

packages = \
[package]

package_data = \
{'': ['*']}

install_requires = \
[]

