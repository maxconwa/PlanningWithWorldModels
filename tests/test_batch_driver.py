import elements
import embodied
import numpy as np

from environments.batch_driver import BatchDriver


class Counter(embodied.Env):
  """Each world counts up from 0 and ends after `length` steps."""

  BATCHED = True

  def __init__(self, batch=1, length=3):
    self.batch, self.length = batch, length
    self.t = np.zeros(batch, int)
    self.done = np.ones(batch, bool)

  @property
  def obs_space(self):
    return {'x': elements.Space(np.float32),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool), 'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool)}

  @property
  def act_space(self):
    return {'reset': elements.Space(bool),
            'action': elements.Space(np.float32, (2,), -1, 1)}

  def step_batch(self, acts):
    restart = acts['reset'] | self.done
    self.t = np.where(restart, 0, self.t + 1)
    self.done = self.t >= self.length
    return {'x': self.t.astype(np.float32),
            'reward': np.ones(self.batch, np.float32),
            'is_first': restart.copy(), 'is_last': self.done.copy(),
            'is_terminal': self.done.copy()}

  def step(self, action):
    acts = {k: np.asarray(v)[None] for k, v in action.items()}
    return {k: v[0] for k, v in self.step_batch(acts).items()}


def policy(carry, obs):
  return carry, {'action': np.zeros((len(obs['x']), 2), np.float32)}, {}


def record(driver, steps):
  seen = []
  driver.on_step(lambda tran, i: seen.append(
      (i, float(tran['x']), bool(tran['is_first']))))
  driver.reset()
  driver(policy, steps=steps)
  return sorted(seen)


def test_matches_dreamers_driver():
  ours = record(BatchDriver([lambda **kw: Counter(**kw)] * 3), 12)
  theirs = record(embodied.Driver([lambda: Counter()] * 3, parallel=False), 12)
  assert ours == theirs


def test_builds_one_batched_env():
  calls = []

  def make(**kw):
    calls.append(kw)
    return Counter(**kw)

  BatchDriver([make] * 5)
  assert calls == [{'batch': 5}]
