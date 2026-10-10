import jax
import jax.numpy as jnp
import numpy as np
import pytest

import environments
from environments.stack_options import (
    INVALID_STEPS, NUM_OPTIONS, BatchStackOptions, obs_space, option_mask)
from environments.stack_scene import HALF, PARK
from environments.stack_task import LIMITS

B = 4
TABLE = 8
NOBODY = jnp.zeros(4, bool)


def pos(**blocks):
  """Block positions: absent blocks parked, the named ones at (x, y, z)."""
  out = np.array(PARK, np.float32)
  for k, xyz in blocks.items():
    out['rgby'.index(k)] = xyz
  return jnp.asarray(out)


def acts(option, reset=False, batch=B):
  return {'reset': np.full(batch, reset),
          'action': np.broadcast_to(np.asarray(option, np.int32), (batch,))}


def teleport(env, world, placements):
  """Move blocks of one world: {colour: (x, y, z)}; others stay put."""
  sim, d = env.base.sim, env.state['data']
  p = np.array(jax.device_get(sim.block_pos(d)))
  for c, xyz in placements.items():
    p[world, c] = xyz
  env.state['data'] = sim.set_blocks(
      d, jax.device_put(p.astype(np.float32)),
      jax.device_put(np.zeros((env.num_envs, 4), np.float32)))


def test_mask_with_an_empty_gripper():
  present = jnp.array([True, True, False, False])
  m = np.asarray(option_mask(pos(r=(0.4, 0, HALF), g=(0.6, 0, HALF)),
                             present, NOBODY))
  assert m.tolist() == [1, 1, 0, 0] + [0] * 4 + [0]


def test_mask_while_holding():
  present = jnp.array([True, True, True, False])
  p = pos(r=(0.5, 0, 0.15), g=(0.4, 0, HALF), b=(0.6, 0, HALF))
  m = np.asarray(option_mask(p, present, NOBODY.at[0].set(True)))
  assert m.tolist() == [0] * 4 + [0, 1, 1, 0] + [1]


def test_mask_skips_covered_blocks():
  present = jnp.array([True, True, False, False])
  p = pos(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF))
  m = np.asarray(option_mask(p, present, NOBODY))
  assert m[:2].tolist() == [0, 1]


def test_a_held_block_does_not_cover_the_one_below():
  present = jnp.array([True, True, False, False])
  p = pos(r=(0.5, 0, HALF), g=(0.5, 0, 0.12))
  m = np.asarray(option_mask(p, present, NOBODY.at[1].set(True)))
  assert m[4] and not m[5]


@pytest.fixture(scope='module')
def env():
  return BatchStackOptions(B, n=2, prestack=0.0, seed=0)


def check_obs(obs, batch=B):
  for key, space in obs_space((64, 64)).items():
    value = np.asarray(obs[key])
    assert value.shape == (batch,) + space.shape, key
    assert value.dtype == space.dtype, key


def test_first_step(env):
  obs = env.step(acts(TABLE, reset=True))
  check_obs(obs)
  assert obs['is_first'].all() and not obs['is_last'].any()
  assert (obs['reward'] == 0).all() and (obs['log/duration'] == 0).all()
  assert env.privileged()['option_mask'].shape == (B, NUM_OPTIONS)


def test_invalid_option_waits(env):
  env.step(acts(TABLE, reset=True))
  # The home pose is above the controller's ceiling, so the arm first sinks
  # to it; hold once to let it settle.
  first = env.step(acts(TABLE))
  before = env.privileged()['blocks']
  obs = env.step(acts(TABLE))
  assert (obs['log/duration'] == INVALID_STEPS).all()
  assert (obs['log/sim_steps'] - first['log/sim_steps'] ==
          INVALID_STEPS).all()
  np.testing.assert_allclose(obs['arm'][:, 14:17], first['arm'][:, 14:17],
                             atol=0.01)
  np.testing.assert_allclose(env.privileged()['blocks'][..., 1:4],
                             before[..., 1:4], atol=0.005)
  assert (obs['reward'] == 0).all()


def test_pick_lifts_the_block(env):
  obs = env.step(acts(TABLE, reset=True))
  colour = obs['goal'][:, 0]
  obs = env.step({'reset': np.zeros(B, bool),
                  'action': colour.astype(np.int32)})
  held = env.privileged()['held']
  assert (held == colour).sum() >= 3
  assert (obs['log/duration'] < 150).sum() >= 3
  assert (obs['log/duration'] > INVALID_STEPS).all()


def test_worlds_wait_for_each_other(env):
  """Odd worlds run an invalid option while even worlds pick: the odd ones
  take exactly INVALID_STEPS and do not move, however long the picks run."""
  first = env.step(acts(TABLE, reset=True))
  before = env.privileged()['blocks']
  waits = np.arange(B) % 2 == 1
  options = np.where(waits, TABLE, first['goal'][:, 0]).astype(np.int32)
  obs = env.step({'reset': np.zeros(B, bool), 'action': options})
  delta = obs['log/sim_steps'] - first['log/sim_steps']
  assert (delta[waits] == INVALID_STEPS).all()
  assert (delta[~waits] > INVALID_STEPS).all()
  np.testing.assert_allclose(env.privileged()['blocks'][waits][..., 1:4],
                             before[waits][..., 1:4], atol=0.005)


def test_invalid_option_keeps_the_grip(env):
  obs = env.step(acts(TABLE, reset=True))
  colour = obs['goal'][:, 0]
  env.step({'reset': np.zeros(B, bool), 'action': colour.astype(np.int32)})
  held = env.privileged()['held']
  env.step(acts(colour[0]))  # pick(...) while holding: invalid.
  after = env.privileged()['held']
  assert ((held >= 0) <= (after == held)).all()


def test_completed_tower_waits_for_success(env):
  obs = env.step(acts(TABLE, reset=True))
  goal = obs['goal'][0]
  teleport(env, 0, {int(goal[0]): (0.5, 0.15, HALF),
                    int(goal[1]): (0.5, 0.15, 3 * HALF)})
  out = [env.step(acts(TABLE)) for _ in range(3)]
  # The first goal block counts once it rests on the table, so the start
  # may already have prefix 1.
  assert [o['reward'][0] for o in out] == [2 - obs['log/prefix'][0], 0, 1]
  assert [o['log/duration'][0] for o in out] == [10, 10, 1]
  assert out[2]['is_terminal'][0] and out[2]['is_last'][0]


def test_time_limit_cuts_the_option(env):
  env.step(acts(TABLE, reset=True))
  steps = np.array(jax.device_get(env.state['steps']))
  steps[:] = LIMITS[2] - 3
  env.state['steps'] = jax.device_put(steps.astype(np.int32))
  obs = env.step(acts(TABLE))
  assert (obs['log/duration'] == 3).all() and obs['is_last'].all()
  assert not obs['is_terminal'].any()


def test_runs_under_transfer_guard(env):
  with jax.transfer_guard('disallow'):
    env.step(acts(TABLE, reset=True))
    env.step(acts(TABLE))
    env.privileged()


def test_registered():
  e = environments.make('stack_options', batch=2, n=2)
  assert e.act_space['action'].discrete
  assert e.act_space['action'].high == NUM_OPTIONS
  assert 'log/duration' in e.obs_space
