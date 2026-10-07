import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import HALF, PARK
from environments.stack_task import (
    HOLD_STEPS, JITTER, NONE, SLOTS, held_mask, progress, sample_start,
    tower_prefix)

NOBODY = jnp.zeros(4, bool)


def tower(goal, xy=(0.5, 0.0)):
  """Block positions for a perfect tower in goal order, others parked."""
  pos = PARK.copy()
  for level, c in enumerate(goal):
    pos[c] = (xy[0], xy[1], HALF + 2 * HALF * level)
  return jnp.asarray(pos)


def padded(goal):
  return jnp.array(list(goal) + [NONE] * (4 - len(goal)))


def test_prefix_of_a_full_tower():
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 0, 1)), NOBODY)) == 3


def test_prefix_stops_at_the_wrong_order():
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 1, 0)), NOBODY)) == 1


def test_prefix_is_zero_when_bottom_block_is_lifted():
  pos = tower((2, 0)).at[2, 2].add(0.05).at[0, 2].add(0.05)
  assert int(tower_prefix(padded((2, 0)), 2, pos, NOBODY)) == 0


def test_prefix_ignores_held_blocks():
  held = NOBODY.at[1].set(True)
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 0, 1)), held)) == 2


def test_prefix_needs_alignment():
  pos = tower((2, 0)).at[0, 0].add(0.02)
  assert int(tower_prefix(padded((2, 0)), 2, pos, NOBODY)) == 1


def test_held_needs_a_block_between_closed_fingers():
  pos = jnp.asarray(PARK).at[3].set(jnp.array([0.5, 0.0, 0.1]))
  site = jnp.array([0.5, 0.0, 0.1])
  assert held_mask(pos, site, 0.04).tolist() == [False, False, False, True]
  assert not held_mask(pos, site, 0.08).any()   # Open.
  assert not held_mask(pos, site, 0.0).any()    # Closed on nothing.


def test_progress_rewards_changes_in_prefix():
  r, s, count = progress(1, 0, 2, 3, False)
  assert (float(r), bool(s)) == (1.0, False)
  r, s, count = progress(2, count, 1, 3, False)
  assert float(r) == -1.0


def test_success_needs_the_hold():
  k_old, count = 1, 0
  rewards, successes = [], []
  for _ in range(HOLD_STEPS + 1):
    r, s, count = progress(k_old, count, 2, 2, True)
    k_old = 2
    rewards.append(float(r))
    successes.append(bool(s))
  assert rewards[0] == 1.0 and not any(successes[:-1])
  assert successes[-1] and rewards[-1] == 1.0


def test_hold_restarts_when_gripper_closes():
  count = 0
  for _ in range(HOLD_STEPS - 1):
    _, _, count = progress(2, count, 2, 2, True)
  _, _, count = progress(2, count, 2, 2, False)
  assert int(count) == 0


def test_starts_are_separated_and_in_the_workspace():
  keys = jax.random.split(jax.random.PRNGKey(0), 256)
  n, goal, pos, _ = jax.vmap(sample_start, (0, None, None))(keys, 0, 0.0)
  n, goal, pos = map(np.asarray, (n, goal, pos))
  assert set(n.tolist()) == {2, 3, 4}
  lo, hi = SLOTS.min(0) - JITTER, SLOTS.max(0) + JITTER
  for i in range(len(n)):
    blocks = goal[i][:n[i]]
    assert (goal[i][n[i]:] == NONE).all()
    xy = pos[i][blocks][:, :2]
    assert ((xy >= lo) & (xy <= hi)).all()
    gaps = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    assert gaps[np.triu_indices(len(blocks), 1)].min() >= 0.07
    absent = [c for c in range(4) if c not in blocks]
    np.testing.assert_allclose(pos[i][absent], PARK[absent])


def test_prestacked_starts_stack_two_or_three():
  keys = jax.random.split(jax.random.PRNGKey(1), 256)
  n, goal, pos, _ = jax.vmap(sample_start, (0, None, None))(keys, 0, 1.0)
  n, goal, pos = map(np.asarray, (n, goal, pos))
  for i in range(len(n)):
    zs = pos[i][goal[i][:n[i]]][:, 2]
    assert int((zs > HALF + 1e-4).sum()) + 1 in (2, 3)


def test_fixed_n():
  keys = jax.random.split(jax.random.PRNGKey(2), 32)
  n, *_ = jax.vmap(sample_start, (0, None, None))(keys, 3, 0.3)
  assert set(np.asarray(n).tolist()) == {3}
