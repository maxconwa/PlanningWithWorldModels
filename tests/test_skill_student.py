import json

import jax
import jax.numpy as jnp
import numpy as np

from skills import scripted
from skills.policy import load_student
from skills.student import Student, act, dagger


def test_student_outputs_actions_in_range():
  net = Student()
  images = jnp.zeros((3, 64, 64, 6), jnp.uint8)
  proprio = jnp.zeros((3, 25))
  params = net.init(jax.random.PRNGKey(0), images, proprio)
  out = act(params, {'workspace': images[..., :3], 'wrist': images[..., 3:],
                     'proprio': proprio})
  assert out.shape == (3, 5)
  assert np.abs(np.asarray(out)).max() <= 1.0


def test_dagger_fits_the_teacher(tmp_path):
  config, _ = dagger('pick', scripted.pick, tmp_path, iterations=3,
                     num_envs=8, rollout=40, stride=2, train_steps=150,
                     capacity=2000, batch=64)
  losses = [json.loads(line)['loss']
            for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
  assert losses[-1] < losses[0]
  assert config['control_steps'] == 3 * 8 * 40
  assert (tmp_path / 'params.pkl').exists()


def test_saved_student_reloads(tmp_path):
  dagger('pick', scripted.pick, tmp_path, iterations=1, num_envs=4,
         rollout=10, stride=1, train_steps=5, capacity=100, batch=8)
  policy = load_student(tmp_path)
  obs = {'workspace': jnp.zeros((2, 64, 64, 3), jnp.uint8),
         'wrist': jnp.full((2, 64, 64, 3), 128, jnp.uint8),
         'proprio': jnp.ones((2, 25))}
  np.testing.assert_allclose(np.asarray(policy(obs)),
                             np.asarray(load_student(tmp_path)(obs)))


def test_scripted_teacher_labels_states():
  """Students can learn from the scripted controllers (the stage 2 teachers
  until PPO teachers pass their gate)."""
  from skills.student import teacher_policy
  state = jnp.zeros((4, 85)).at[:, 75].set(1.0)
  for skill in ('pick', 'place'):
    labels = teacher_policy(skill, 'scripted')(state)
    assert labels.shape == (4, 5)
    np.testing.assert_allclose(
        np.asarray(labels),
        np.asarray(getattr(scripted, skill)(state)))


def test_saved_student_matches_the_trained_one(tmp_path):
  """The pickled parameters are the ones training ended with."""
  config, params = dagger('pick', scripted.pick, tmp_path, iterations=1,
                          num_envs=4, rollout=10, stride=1, train_steps=5,
                          capacity=100, batch=8)
  obs = {'workspace': jnp.full((2, 64, 64, 3), 30, jnp.uint8),
         'wrist': jnp.full((2, 64, 64, 3), 200, jnp.uint8),
         'proprio': jnp.linspace(-1, 1, 50).reshape(2, 25)}
  np.testing.assert_allclose(np.asarray(load_student(tmp_path)(obs)),
                             np.asarray(act(params, obs)), atol=1e-6)
  assert config['teacher'] == 'scripted'


def test_student_params_run_under_dreamers_transfer_guard(tmp_path):
  """Stage 4 passes the parameters into its jitted step as an argument."""
  from skills.policy import student_params
  dagger('pick', scripted.pick, tmp_path, iterations=1, num_envs=4,
         rollout=10, stride=1, train_steps=5, capacity=100, batch=8)
  params = student_params(tmp_path)
  obs = {'workspace': jnp.zeros((2, 64, 64, 3), jnp.uint8),
         'wrist': jnp.zeros((2, 64, 64, 3), jnp.uint8),
         'proprio': jnp.zeros((2, 25))}
  step = jax.jit(act)
  with jax.transfer_guard('disallow'):
    assert step(params, obs).shape == (2, 5)


def test_student_proprio_layout():
  from skills.env import PROPRIO_DIM, student_proprio
  arm = jnp.arange(19.0)[None].repeat(3, 0)
  out = student_proprio(arm, jnp.array([0, 2, 4]), jnp.array([0, 75, 150]))
  assert out.shape == (3, PROPRIO_DIM)
  np.testing.assert_allclose(np.asarray(out[:, :19]), np.asarray(arm))
  assert np.asarray(out[2, 19:24]).tolist() == [0, 0, 0, 0, 1]
  np.testing.assert_allclose(np.asarray(out[:, 24]), [0.0, 0.5, 1.0], atol=1e-6)


def test_student_defaults_reproduce_the_gated_students():
  import inspect
  from skills import student
  assert inspect.signature(student.dagger).parameters['iterations'].default == 30
