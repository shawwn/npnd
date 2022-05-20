# -*- coding: utf-8 -*-
from setuptools import setup
import os

def path(to):
  return os.path.join(os.path.dirname(__file__), to)

exec(compile(open(path("setup_info.py")).read(), path("setup_info.py"), "exec"))

package_dir = \
{'': 'src'}

packages = \
['npnd', 'npnd._src', 'npnd._src.numpy', 'npnd.numpy', 'npnd.ops']

package_data = \
{'': ['*']}

install_requires = \
['pytreez>=1.4,<2.0']


setup_kwargs = {
    'package_dir': package_dir,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
    **(globals().get('base_kwargs')),
}


setup(**setup_kwargs)
