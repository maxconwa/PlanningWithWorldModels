import numpy as np

from environments.stack_scene import HALF
from skills.tower import choose

TABLE = 8


def blocks(**xyz):
  out = np.zeros((4, 11))
  for c, k in enumerate('rgby'):
    if k in xyz:
      out[c, 0] = 1
      out[c, 1:4] = xyz[k]
  return out


def mask(b, held):
  present = b[:, 0] > 0
  clear = present.copy()
  for c in range(4):
    for o in range(4):
      if (o != c and present[o] and o != held and
          np.linalg.norm(b[o, 1:3] - b[c, 1:3]) < 0.03 and
          b[o, 3] > b[c, 3] + HALF):
        clear[c] = False
  clear[held] = False if held >= 0 else clear[held]
  holding = held >= 0
  return np.concatenate([clear & ~holding, clear & holding, [holding]])


GOAL = np.array([0, 1, 2, 4])  # red, green, blue bottom to top.


def test_picks_the_next_goal_block():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.6, 0, HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 1) == 1


def test_places_the_next_block_on_the_tower():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, 0.12), b=(0.6, 0, HALF))
  assert choose(b, 1, mask(b, 1), GOAL, 3, 1) == 4 + 0


def test_puts_a_wrong_block_on_the_table():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.6, 0, 0.12))
  assert choose(b, 2, mask(b, 2), GOAL, 3, 1) == TABLE


def test_clears_a_block_off_the_tower():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.4, 0, 3 * HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 1) == 2


def test_moves_a_misplaced_base_to_the_table():
  b = blocks(r=(0.5, 0, 3 * HALF), g=(0.5, 0, HALF), b=(0.6, 0, HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 0) == 0
  assert choose(b, 0, mask(b, 0), GOAL, 3, 0) == TABLE


def test_waits_once_the_tower_is_built():
  b = blocks(r=(0.4, 0, HALF), g=(0.4, 0, 3 * HALF), b=(0.4, 0, 5 * HALF))
  option = choose(b, -1, mask(b, -1), GOAL, 3, 3)
  assert not mask(b, -1)[option]
