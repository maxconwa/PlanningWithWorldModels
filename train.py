"""Train DreamerV3, with the environments in environments/ as the custom suite.

Takes the same flags as third_party/dreamerv3/dreamerv3/main.py, which it runs;
every built-in task still works. Custom environments are --task custom_<name>:

  python train.py --task custom_catch --configs debug --logdir /tmp/catch
  python train.py --task custom_tetris --configs atari --env.atari.gray False
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

import jax_compat  # noqa: E402,F401
from dreamerv3 import main  # noqa: E402
import dreamer_fixes  # noqa: E402,F401
import embodied  # noqa: E402
from environments import batch_driver  # noqa: E402

builtin_make_env = main.make_env


BATCHED = False


def make_env(config, index, **overrides):
  global BATCHED
  suite, task = config.task.split('_', 1)
  if suite != 'custom':
    return builtin_make_env(config, index, **overrides)
  import environments
  ctor = environments.lookup(task)
  BATCHED = getattr(ctor, 'BATCHED', False)
  options = config.env.get(getattr(ctor, 'OPTIONS', 'custom'), {})
  kwargs = {**options, **overrides}
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  return main.wrap_env(ctor(task, **kwargs), config)


# main.main() looks make_env up by name when it builds the environments, and
# DreamerV3's run scripts look embodied.Driver up when they make the driver.
# The agent is built first, through make_env, so BATCHED is set by then.
main.make_env = make_env
embodied.Driver = batch_driver.make_driver(embodied.Driver, lambda: BATCHED)

if __name__ == '__main__':
  main.main()
