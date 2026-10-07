"""Episode bookkeeping for Brax PPO with environments that reset their own
worlds.

Brax's EpisodeWrapper and AutoResetWrapper keep these fields, but its
auto-reset returns every world to its first start. The skill environments
draw a new start whenever a world's episode ends, so they skip Brax's
wrappers (ppo.train(..., wrap_env=False)) and keep the same fields here.
"""

import jax.numpy as jnp


def episode_info(batch, metric_names):
  """The fields Brax PPO reads from state.info, for a fresh batch."""
  zeros = jnp.zeros(batch, jnp.float32)
  metrics = {'sum_reward': zeros, 'length': zeros}
  metrics.update({name: zeros for name in metric_names})
  return {'truncation': zeros, 'episode_done': zeros,
          'episode_metrics': metrics}


def track_episode(info, reward, done, truncation, metrics):
  """Adds one step to each world's episode totals, as EpisodeWrapper does;
  totals restart on the step after a world's episode ended."""
  keep = 1.0 - info['episode_done']
  totals = {k: v * keep for k, v in info['episode_metrics'].items()}
  totals['sum_reward'] = totals['sum_reward'] + reward
  totals['length'] = totals['length'] + 1.0
  for name, value in metrics.items():
    totals[name] = totals[name] + value
  return dict(info, truncation=truncation.astype(jnp.float32),
              episode_done=done.astype(jnp.float32), episode_metrics=totals)
