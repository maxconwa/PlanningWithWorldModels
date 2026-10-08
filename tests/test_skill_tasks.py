import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import HALF, PARK
from skills.tasks import (
    LIFT, SITE_HI, SITE_LO, TABLE, clear_mask, clearance, pick_progress,
    pick_start, place_progress, place_start, resting_on)

ALL = jnp.ones(4, bool)
NOBODY = jnp.zeros(4, bool)


def layout(**blocks):
  """Positions with the named colours (r, g, b, y) placed, others parked."""
  pos = PARK.copy()
  for name, xyz in blocks.items():
    pos['rgby'.index(name)] = xyz
  return jnp.asarray(pos)


def test_clear_blocks_have_nothing_on_top():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF), b=(0.4, 0.1, HALF))
  present = jnp.array([True, True, True, False])
  assert clear_mask(pos, present).tolist() == [False, True, True, False]


def test_clearance_measures_the_gap_below():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF + 0.03))
  present = jnp.array([True, True, False, False])
  assert abs(float(clearance(pos, present, 1)) - 0.03) < 1e-5
  assert abs(float(clearance(pos, present, 0))) < 1e-6


def test_pick_succeeds_only_when_held_and_lifted():
  pos = layout(r=(0.5, 0, HALF + LIFT + 0.01))
  present = jnp.array([True, False, False, False])
  site = pos[0]
  _, ok = pick_progress(pos, present, NOBODY.at[0].set(True), site, 0)
  assert bool(ok)
  _, ok = pick_progress(pos, present, NOBODY, site, 0)
  assert not bool(ok)
  low = layout(r=(0.5, 0, HALF + 0.02))
  _, ok = pick_progress(low, present, NOBODY.at[0].set(True), low[0], 0)
  assert not bool(ok)


def test_pick_potential_grows_as_the_gripper_nears():
  pos = layout(r=(0.5, 0, HALF))
  present = jnp.array([True, False, False, False])
  far, _ = pick_progress(pos, present, NOBODY, jnp.array([0.5, 0, 0.3]), 0)
  near, _ = pick_progress(pos, present, NOBODY, jnp.array([0.5, 0, 0.1]), 0)
  assert float(near) > float(far)


def test_resting_on_a_block_and_on_the_table():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0.005, 3 * HALF))
  assert bool(resting_on(pos, 1, 0))
  assert not bool(resting_on(pos, 1, TABLE))
  assert bool(resting_on(pos, 0, TABLE))


def test_place_succeeds_when_released_open_and_raised():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF))
  top = 3 * HALF + HALF
  site = jnp.array([0.5, 0, top + 0.06])
  _, ok = place_progress(pos, NOBODY, site, 0.08, 1, 0)
  assert bool(ok)
  _, ok = place_progress(pos, NOBODY.at[1].set(True), site, 0.04, 1, 0)
  assert not bool(ok)
  _, ok = place_progress(pos, NOBODY, site.at[2].set(top + 0.02), 0.08, 1, 0)
  assert not bool(ok)


def test_pick_targets_are_present_and_clear():
  keys = jax.random.split(jax.random.PRNGKey(0), 512)
  present, pos, _, target = jax.vmap(pick_start, (0, None))(keys, 0.5)
  for i in range(len(keys)):
    assert bool(clear_mask(pos[i], present[i])[target[i]])


def test_place_starts_put_the_block_in_reach_above_everything():
  keys = jax.random.split(jax.random.PRNGKey(1), 512)
  present, pos, _, held, target, site, _ = jax.vmap(
      place_start, (0, None))(keys, 0.5)
  pos, held, target, site = map(np.asarray, (pos, held, target, site))
  for i in range(len(keys)):
    np.testing.assert_allclose(pos[i, held[i]], site[i])
    assert (site[i] >= np.array(SITE_LO) - 1e-6).all()
    assert (site[i] <= np.array(SITE_HI) + 1e-6).all()
    others = [c for c in range(4) if present[i, c] and c != held[i]]
    assert pos[i, held[i], 2] - HALF > max(pos[i, others, 2] + HALF)
    assert target[i] != held[i]
    if target[i] != TABLE:
      assert present[i, target[i]]


def test_pick_potential_keeps_rising_past_the_success_lift():
  """The reward keeps paying a little past LIFT, so a policy ends the skill
  safely clear of the success line rather than on it."""
  present = jnp.array([True, False, False, False])
  held = NOBODY.at[0].set(True)
  at_line = layout(r=(0.5, 0, HALF + LIFT))
  past = layout(r=(0.5, 0, HALF + LIFT + 0.015))
  phi_line, _ = pick_progress(at_line, present, held, at_line[0], 0)
  phi_past, _ = pick_progress(past, present, held, past[0], 0)
  assert float(phi_past) > float(phi_line) + 0.1


def test_pick_starts_can_favour_stacked_targets():
  keys = jax.random.split(jax.random.PRNGKey(4), 1024)
  present, pos, _, target = jax.vmap(pick_start, (0, None, None))(
      keys, 1.0, 0.8)
  z = np.asarray(pos)[np.arange(1024), np.asarray(target), 2]
  assert (z > HALF + 1e-3).mean() >= 0.7
  for i in range(64):
    assert bool(clear_mask(pos[i], present[i])[target[i]])
