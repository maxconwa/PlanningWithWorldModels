import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skills import scripted
from skills.env import ACTION_DIM, PROPRIO_DIM, STATE_DIM, SkillEnv
from skills.tasks import EPISODE, FAIL_PENALTY


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
  assert env.action_size == ACTION_DIM
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


def test_stopping_without_success_is_penalised():
  env, state, step = start('pick', 8)
  stop = jnp.zeros((8, ACTION_DIM)).at[:, 5].set(1.0)
  state = step(state, stop)
  assert (np.asarray(state.done) == 1).all()
  assert (np.asarray(state.metrics['success']) == 0).all()
  assert (np.asarray(state.reward) < -FAIL_PENALTY + 0.2).all()
  assert (np.asarray(state.info['steps']) == 0).all()  # Restarted.


def test_timeout_truncates():
  env, state, step = start('pick', 4)
  hold = jnp.zeros((4, ACTION_DIM)).at[:, 4:].set(-1.0)
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
