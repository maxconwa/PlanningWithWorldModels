"""Custom training environments for world models.

Each entry maps a name to an embodied.Env class as 'module:Class'. The class is
imported only when the environment is made, so an environment's dependencies
are needed only by the runs that use it. Train on one with train.py and
--task custom_<name>.

A class takes its options from the config section named by its OPTIONS
attribute (config.env.custom by default), so an environment built on a
DreamerV3 suite can share that suite's options.
"""

import importlib

REGISTRY = {
    'catch': 'environments.catch:Catch',
    'tetris': 'environments.tetris:Tetris',
}


def lookup(name):
  """The environment class registered as `name`."""
  if name not in REGISTRY:
    raise KeyError(f'Unknown custom environment {name!r}; '
                   f'registered: {", ".join(sorted(REGISTRY))}')
  module, cls = REGISTRY[name].split(':')
  return getattr(importlib.import_module(module), cls)


def make(name, **kwargs):
  return lookup(name)(name, **kwargs)
