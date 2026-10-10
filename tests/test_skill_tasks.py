import jax
import jax.numpy as jnp
import numpy as np
import pytest

from environments.stack_scene import HALF, PARK
from skills.tasks import (
    ARM_HI, ARM_LO, FREE, LIFT, RAISE, SITE_HI, TABLE, arm_start, clear_mask,
    clearance, pick_progress, pick_start, place_progress, place_start,
    resting_on, table_spot)

ALL = jnp.ones(4, bool)
NOBODY = jnp.zeros(4, bool)
PRESENT = jnp.array([True, True, False, False])


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
  _, ok = place_progress(pos, PRESENT, NOBODY, site, 0.08, 1, 0)
  assert bool(ok)
  _, ok = place_progress(pos, PRESENT, NOBODY.at[1].set(True), site, 0.04, 1, 0)
  assert not bool(ok)
  _, ok = place_progress(pos, PRESENT, NOBODY, site.at[2].set(top + 0.02), 0.08, 1, 0)
  assert not bool(ok)


def test_pick_targets_are_present_and_clear():
  keys = jax.random.split(jax.random.PRNGKey(0), 512)
  present, pos, _, target = jax.vmap(pick_start, (0, None))(keys, 0.5)
  for i in range(len(keys)):
    assert bool(clear_mask(pos[i], present[i])[target[i]])


def test_place_starts_hold_the_block_clear_of_the_blocks_under_it():
  keys = jax.random.split(jax.random.PRNGKey(1), 512)
  present, pos, _, held, target, site, _ = jax.vmap(
      place_start, (0, None))(keys, 0.5)
  present, pos, held, target, site = map(
      np.asarray, (present, pos, held, target, site))
  low = 0
  for i in range(len(keys)):
    np.testing.assert_allclose(pos[i, held[i]], site[i])
    assert (site[i, :2] >= np.array(ARM_LO) - 1e-6).all()
    assert (site[i, :2] <= np.array(ARM_HI) + 1e-6).all()
    others = [c for c in range(4) if present[i, c] and c != held[i]
              and np.linalg.norm(pos[i, c, :2] - site[i, :2]) < FREE]
    under = max([0.0] + [pos[i, c, 2] + HALF for c in others])
    assert site[i, 2] - HALF >= under + 0.01 - 1e-6
    assert site[i, 2] <= max(SITE_HI[2], under + 2 * HALF + 0.01) + 1e-6
    low += site[i, 2] < 0.12
    assert target[i] != held[i]
    if target[i] != TABLE:
      assert present[i, target[i]]
  assert low > 50  # Some starts hold the block low, as after a pick.


def test_table_spot_is_free_and_on_the_table():
  keys = jax.random.split(jax.random.PRNGKey(5), 256)
  present, pos, _, held, _, _, _ = jax.vmap(
      place_start, (0, None))(keys, 1.0)
  spots = jax.vmap(table_spot)(pos, present, held)
  pos, present, held, spots = map(np.asarray, (pos, present, held, spots))
  for i in range(len(keys)):
    assert spots[i, 2] == pytest.approx(HALF)
    for c in range(4):
      if present[i, c] and c != held[i]:
        assert np.linalg.norm(spots[i, :2] - pos[i, c, :2]) >= FREE - 1e-6


def test_table_spot_is_under_the_block_when_that_is_free():
  pos = layout(r=(0.45, -0.05, 0.15), g=(0.65, 0.15, HALF))
  spot = np.asarray(table_spot(pos, PRESENT, 0))
  np.testing.assert_allclose(spot, [0.45, -0.05, HALF], atol=1e-6)


def test_table_spot_moves_off_a_block_below():
  pos = layout(r=(0.45, -0.05, 0.15), g=(0.45, -0.05, HALF))
  spot = np.asarray(table_spot(pos, PRESENT, 0))
  assert np.linalg.norm(spot[:2] - np.array([0.45, -0.05])) >= FREE


def test_arm_starts_are_clear_of_the_blocks():
  keys = jax.random.split(jax.random.PRNGKey(6), 512)
  present, pos, _, _ = jax.vmap(pick_start, (0, None))(keys, 1.0)
  site, yaw = jax.vmap(arm_start)(keys, pos, present)
  pos, present, site = map(np.asarray, (pos, present, site))
  assert (site[:, :2] >= np.array(ARM_LO) - 1e-6).all()
  assert (site[:, :2] <= np.array(ARM_HI) + 1e-6).all()
  assert (np.abs(np.asarray(yaw)) <= np.pi / 4 + 1e-6).all()
  for i in range(len(keys)):
    near = present[i] & (np.linalg.norm(
        pos[i, :, :2] - site[i, :2], axis=-1) < FREE)
    top = max([0.0] + list(pos[i, near, 2] + HALF))
    assert site[i, 2] >= top + RAISE - 1e-6


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
