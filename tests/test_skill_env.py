import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skills import scripted
from skills.env import ACTION_DIM, PROPRIO_DIM, STATE_DIM, SkillEnv
from skills.tasks import EPISODE, SUCCESS_BONUS


def start(skill, batch, cameras=False, seed=0):
  env = SkillEnv(skill, batch, cameras=cameras)
  keys = jax.random.split(jax.random.PRNGKey(seed), batch)
  return env, jax.jit(env.reset)(keys), jax.jit(env.step)


def run(env, state, step, policy, steps=EPISODE):
  """Steps until every world has ended once; first-episode success."""
  ended = np.zeros(env.num_envs, bool)
  success = np.zeros(env.num_envs, bool)
  for _ in range(steps):
    state = step(state, policy(state.obs['state']))
    done = np.asarray(state.done) > 0
    first = done & ~ended
    success[first] = np.asarray(state.metrics['success'])[first] > 0
    ended |= done
    if ended.all():
      break
  return success, ended


@pytest.mark.parametrize('skill', ['pick', 'place'])
def test_reset_shapes(skill):
  env, state, _ = start(skill, 8)
  assert state.obs['state'].shape == (8, STATE_DIM)
  assert env.action_size == ACTION_DIM == 5
  assert np.isfinite(np.asarray(state.obs['state'])).all()


def test_cameras_add_student_inputs():
  _, state, _ = start('pick', 4, cameras=True)
  assert state.obs['workspace'].shape == (4, 64, 64, 3)
  assert state.obs['workspace'].dtype == jnp.uint8
  assert state.obs['proprio'].shape == (4, PROPRIO_DIM)


def test_place_start_holds_the_block():
  env, state, _ = start('place', 64)
  held = np.asarray(env.held(state))
  block = np.asarray(state.info['task']['held_block'])
  assert held[np.arange(64), block].mean() >= 0.95


def test_success_ends_the_skill_with_the_bonus():
  """The environment, not the policy, ends a skill: on the step its
  success condition holds, or at the timeout."""
  env, state, step = start('pick', 32, seed=3)
  successes = 0
  for _ in range(EPISODE):
    state = step(state, scripted.pick(state.obs['state']))
    done = np.asarray(state.done) > 0
    success = np.asarray(state.metrics['success']) > 0
    truncated = np.asarray(state.info['truncation']) > 0
    assert (done == (success | truncated)).all()
    assert (np.asarray(state.reward)[success] > SUCCESS_BONUS - 0.5).all()
    successes += success.sum()
  assert successes > 0


def test_timeout_truncates():
  env, state, step = start('pick', 4)
  hold = jnp.zeros((4, ACTION_DIM)).at[:, 4].set(-1.0)
  for _ in range(EPISODE):
    state = step(state, hold)
  assert (np.asarray(state.done) == 1).all()
  assert (np.asarray(state.info['truncation']) == 1).all()


def test_scripted_pick_succeeds():
  env, state, step = start('pick', 64, seed=1)
  success, ended = run(env, state, step, scripted.pick)
  assert ended.all()
  assert success.mean() >= 0.8, success.mean()


def test_scripted_place_succeeds():
  env, state, step = start('place', 64, seed=2)
  success, ended = run(env, state, step, scripted.place)
  assert ended.all()
  assert success.mean() >= 0.8, success.mean()


def test_pick_env_can_favour_stacked_targets():
  env = SkillEnv('pick', 64, prestack=1.0, stacked=0.8)
  state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(5), 64))
  pos = np.asarray(env.sim.block_pos(state.pipeline_state))
  target = np.asarray(state.info['task']['target'])
  assert (pos[np.arange(64), target, 2] > 0.03).mean() >= 0.7


def test_a_restart_leaves_the_other_worlds_alone():
  """Restarting one world must not change how the others evolve; otherwise
  success depends on when other episodes end and on the batch size."""
  def trajectory(restart_world_0):
    env, state, step = start('pick', 8, seed=6)
    for _ in range(15):
      state = step(state, scripted.pick(state.obs['state']))
    if restart_world_0:
      mask = jnp.zeros(8, bool).at[0].set(True)
      d, info = jax.jit(env._restart)(state.pipeline_state, state.info, mask)
      state = state.replace(pipeline_state=d, info=info)
    seen, done = [], []
    for _ in range(30):
      state = step(state, scripted.pick(state.obs['state']))
      seen.append(np.asarray(state.obs['state'])[1:])
      done.append(np.asarray(state.done)[1:])
    return np.array(seen), np.array(done)
  (seen_a, done_a), (seen_b, done_b) = trajectory(True), trajectory(False)
  np.testing.assert_array_equal(done_a, done_b)
  # Float32 rounding from forwarding twice grows to a few mm through
  # contacts; the bugs this guards against moved worlds by 10 cm or more.
  np.testing.assert_allclose(seen_a, seen_b, atol=2e-2)
