"""Catch: a ball falls down a grid and a paddle on the bottom row catches it.

Small enough to train in minutes, which makes it a quick check that a world
model learns anything at all. Each episode drops --balls balls one after
another; catching one gives +1 and missing it -1.
"""

import elements
import embodied
import numpy as np


class Catch(embodied.Env):

  def __init__(self, task, grid=8, size=(64, 64), balls=10, seed=None):
    del task
    self.grid = grid
    self.size = size
    self.balls = balls
    self.rng = np.random.default_rng(seed)
    self.done = True

  @property
  def obs_space(self):
    return {
        'image': elements.Space(np.uint8, self.size + (3,)),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        # Move the paddle left, keep it still, or move it right.
        'action': elements.Space(np.int32, (), 0, 3),
    }

  def step(self, action):
    if action['reset'] or self.done:
      self.paddle = self.grid // 2
      self.dropped = 0
      self.done = False
      self._drop()
      return self._obs(0.0, is_first=True)
    self.paddle = int(np.clip(self.paddle + action['action'] - 1,
                              0, self.grid - 1))
    self.ball[0] += 1
    reward = 0.0
    if self.ball[0] == self.grid - 1:
      reward = 1.0 if self.ball[1] == self.paddle else -1.0
      if self.dropped == self.balls:
        self.done = True
      else:
        self._drop()
    return self._obs(reward, is_last=self.done, is_terminal=self.done)

  def _drop(self):
    self.ball = [0, int(self.rng.integers(self.grid))]
    self.dropped += 1

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    cells = np.zeros((self.grid, self.grid, 3), np.uint8)
    cells[self.ball[0], self.ball[1]] = (255, 255, 255)
    cells[self.grid - 1, self.paddle] = (0, 160, 255)
    rows, cols = self.size[0] // self.grid, self.size[1] // self.grid
    image = cells.repeat(rows, 0).repeat(cols, 1)
    return dict(
        image=image, reward=np.float32(reward), is_first=is_first,
        is_last=is_last, is_terminal=is_terminal)
