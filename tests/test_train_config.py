import elements
import pytest

import train


@pytest.fixture
def config():
  def parse(*flags):
    configs = train.ConfigYaml(typ='safe').load(
        (train.DREAMER / 'dreamerv3' / 'configs.yaml').read_text())
    return elements.Flags(elements.Config(configs['defaults'])).parse(
        ['--task', 'custom_stack', *flags])
  yield parse
  train.BATCHED = False


def test_stack_options_have_defaults(config):
  assert dict(config().env.stack) == dict(n=0, prestack=0.3)


def test_stack_options_reach_the_env(config):
  env = train.make_env(config('--env.stack.n', '2',
                              '--env.stack.prestack', '0.5'), 0)
  while hasattr(env, 'env') and not hasattr(env, 'options'):
    env = env.env
  assert env.options == dict(n=2, prestack=0.5)


def test_profiler_is_off_and_settable(config):
  # Tracing the GPU beside MuJoCo Warp fails with CUDA_ERROR_LAUNCH_FAILED.
  assert config().jax.profiler is False
  assert config('--jax.profiler', 'True').jax.profiler is True
