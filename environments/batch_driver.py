"""A DreamerV3 driver that steps one batched environment.

embodied.Driver steps environments one at a time. Environments with
BATCHED = True step all their worlds at once on the GPU, so BatchDriver
makes one of them with batch=<number of environments> and calls its
step_batch(). It keeps embodied.Driver's interface and transitions, so the
rest of DreamerV3's training loop is unchanged.
"""

import elements
import numpy as np


class BatchDriver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    del parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    env = make_env_fns[0](batch=self.length)
    # DreamerV3's wrappers keep the environment they wrap as .env; their
    # action normalisation and clipping are identities for [-1, 1] actions.
    while not hasattr(env, 'step_batch'):
      env = env.env
    self.env = env
    self.act_space = env.act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    self.env.close()

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    obs = self.env.step_batch(self.acts)
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)


def make_driver(original, batched):
  """A Driver factory: BatchDriver when `batched()` says the environment
  is batched, `original` otherwise."""

  def driver(make_env_fns, parallel=True, **kwargs):
    if batched():
      return BatchDriver(make_env_fns, parallel, **kwargs)
    return original(make_env_fns, parallel, **kwargs)

  return driver
