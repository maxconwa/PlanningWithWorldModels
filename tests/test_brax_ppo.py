"""Brax PPO trains a batched environment that resets its own worlds."""

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

from skills.brax_util import episode_info, track_episode

LENGTH = 20


class Point(Env):
  """Worlds move a point by 0.1 * action; reward is -|x|. Each world
  restarts at a random x after LENGTH steps."""

  def __init__(self, batch):
    self.batch = batch

  def reset(self, rng):
    x = jax.random.uniform(rng[0], (self.batch,), minval=-1, maxval=1)
    info = dict(episode_info(self.batch, ['size']), rng=rng[0],
                steps=jnp.zeros(self.batch))
    return State(None, {'state': x[:, None]}, jnp.zeros(self.batch),
                 jnp.zeros(self.batch), {'size': jnp.zeros(self.batch)}, info)

  def step(self, state, action):
    x = state.obs['state'][:, 0] + 0.1 * action[:, 0]
    reward = -jnp.abs(x)
    steps = state.info['steps'] + 1
    done = (steps >= LENGTH).astype(jnp.float32)
    info = track_episode(state.info, reward, done, done,
                         {'size': jnp.abs(x)})
    rng, key = jax.random.split(info['rng'])
    fresh = jax.random.uniform(key, (self.batch,), minval=-1, maxval=1)
    x = jnp.where(done > 0, fresh, x)
    info = dict(info, rng=rng, steps=jnp.where(done > 0, 0.0, steps))
    return state.replace(obs={'state': x[:, None]}, reward=reward,
                         done=done, metrics={'size': jnp.abs(x)}, info=info)

  @property
  def observation_size(self):
    return {'state': (1,)}

  @property
  def action_size(self):
    return 1

  @property
  def backend(self):
    return 'none'


def test_ppo_learns_a_self_resetting_batched_env():
  rewards = []
  ppo.train(
      environment=Point(256), eval_env=Point(64), wrap_env=False,
      num_envs=256, num_eval_envs=64, episode_length=LENGTH,
      num_timesteps=300_000, num_evals=3, batch_size=8,
      num_minibatches=32, unroll_length=10, learning_rate=3e-3,
      network_factory=_networks(),
      progress_fn=lambda step, m: rewards.append(
          float(m['eval/episode_reward'])))
  assert rewards[-1] > rewards[0] + 2.0, rewards


def _networks():
  import functools
  from brax.training.agents.ppo import networks
  return functools.partial(
      networks.make_ppo_networks, policy_hidden_layer_sizes=(32, 32),
      value_hidden_layer_sizes=(32, 32), policy_obs_key='state',
      value_obs_key='state')
