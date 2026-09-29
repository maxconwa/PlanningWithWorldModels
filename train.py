"""Train DreamerV3, with the environments in environments/ as the custom suite.

Takes the same flags as third_party/dreamerv3/dreamerv3/main.py, which it runs;
every built-in task still works. Custom environments are --task custom_<name>:

  python train.py --task custom_catch --configs debug --logdir /tmp/catch
"""

import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
DREAMER = HERE / 'third_party' / 'dreamerv3'
# Environments run in subprocesses, which see PYTHONPATH but not sys.path.
os.environ['PYTHONPATH'] = os.pathsep.join(filter(None, [
    str(HERE), str(DREAMER), os.environ.get('PYTHONPATH')]))
sys.path[:0] = [str(HERE), str(DREAMER)]

from dreamerv3 import main  # noqa: E402

builtin_make_env = main.make_env


def make_env(config, index, **overrides):
  suite, task = config.task.split('_', 1)
  if suite != 'custom':
    return builtin_make_env(config, index, **overrides)
  import environments
  kwargs = {**config.env.get('custom', {}), **overrides}
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  return main.wrap_env(environments.make(task, **kwargs), config)


# main.main() looks make_env up by name when it builds the environments.
main.make_env = make_env

if __name__ == '__main__':
  main.main()
