"""A trained DreamerV3 agent as a world model to plan in.

build loads a run's checkpoint; warmup drives the real emulator to get a
posterior state to plan from. Importing this module puts the DreamerV3
checkout on the path, so dreamerv3 and embodied can be imported after it.
"""

import pathlib
import pickle
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DREAMER = REPO / 'third_party' / 'dreamerv3'
sys.path[:0] = [str(DREAMER), str(REPO)]

import elements  # noqa: E402
import jax  # noqa: E402
import jax_compat  # noqa: E402,F401
import ninjax as nj  # noqa: E402
import numpy as np  # noqa: E402


def logdir_arg(value):
  """Argparse type for --logdir: a run directory, or a run name in outputs/train/."""
  path = pathlib.Path(value)
  if not path.exists():
    path = REPO / 'outputs' / 'train' / path
  return path.resolve()


def eval_path(logdir, out):
  """Where an eval output goes: --out as given if it names a directory, otherwise
  a file of that name in outputs/eval/<run>/."""
  path = pathlib.Path(out)
  if path.parent == pathlib.Path('.'):
    path = REPO / 'outputs' / 'eval' / logdir.name / path
  path.parent.mkdir(parents=True, exist_ok=True)
  return path


def seeder(seed):
  """A numpy generator, plus a function that draws fresh JAX keys from it."""
  rng = np.random.default_rng(seed)
  return rng, lambda: jax.device_put(rng.integers(0, 2 ** 32, 2, np.uint32))


def build(logdir, env_seed=None):
  """Load the checkpoint and return the model with a jitted observe step.

  The returned env is seeded with `env_seed` when given, so its no-op starts
  and sticky actions repeat from run to run."""
  config = elements.Config.load(str(logdir / 'config.yaml'))

  from embodied.jax import internal
  from embodied.jax.agent import Options
  internal.setup(**{k: v for k, v in config.jax.items()
                    if k not in Options.__dataclass_fields__})

  from dreamerv3.agent import Agent, sample
  from dreamerv3.main import make_env

  env = make_env(config, 0, **({} if env_seed is None else {'seed': env_seed}))
  obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
  act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
  # The policy's action i is the i-th entry of the env's ALE action set.
  actions = [env.ACTION_MEANING[int(a)] for a in env.actionset]

  # Build the bare ninjax model, skipping the training runner that would
  # allocate a second copy of the parameters and the optimizer state.
  model = object.__new__(Agent)
  model.__init__(obs_space, act_space, elements.Config(
      **config.agent, logdir=str(logdir), seed=config.seed, jax=config.jax,
      batch_size=config.batch_size, batch_length=config.batch_length,
      replay_context=config.replay_context, report_length=config.report_length,
      replica=config.replica, replicas=config.replicas))

  ckpt = (logdir / 'ckpt' / 'latest').read_text().strip()
  with (logdir / 'ckpt' / ckpt / 'agent.pkl').open('rb') as f:
    params = pickle.load(f)['params']
  params = {k: jax.device_put(v) for k, v in params.items()
            if not k.startswith('opt/')}

  def observe(carry, obs, prevact):
    """One real step: encode the frame and update the posterior state."""
    enc_carry, dyn_carry = carry
    enc_carry, _, tokens = model.enc(
        enc_carry, obs, obs['is_first'], False, single=True)
    dyn_carry, _, feat = model.dyn.observe(
        dyn_carry, tokens, prevact, obs['is_first'], False, single=True)
    return (enc_carry, dyn_carry), sample(model.pol(model.feat2tensor(feat), 1))

  return dict(
      config=config, task=config.task, ckpt=ckpt, model=model, params=params,
      env=env, obs_space=obs_space, actions=actions,
      initial=jax.jit(lambda batch=1: (
          model.enc.initial(batch), model.dyn.initial(batch)), static_argnums=0),
      observe=jax.jit(nj.pure(observe)))


def warmup(parts, steps, seed):
  """Drive the real emulator with the trained policy for `steps` frames.

  Returns the posterior RSSM state after the last frame together with that
  frame's observation, and closes the emulator."""
  params, env, observe = parts['params'], parts['env'], parts['observe']
  carry = parts['initial']()
  prevact = {'action': jax.device_put(np.zeros(1, np.int32))}
  act = {'action': np.int32(0), 'reset': True}
  for _ in range(steps):
    obs = env.step(act)
    batch = {k: jax.device_put(np.asarray(obs[k])[None]) for k in parts['obs_space']}
    _, (carry, prevact) = observe(params, carry, batch, prevact, seed=seed())
    act = {'action': jax.device_get(prevact['action'])[0], 'reset': False}
  env.close()
  return carry[1], obs
