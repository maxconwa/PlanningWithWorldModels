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
  config = dagger('pick', scripted.pick, tmp_path, iterations=3, num_envs=8,
                  rollout=40, stride=2, train_steps=150, capacity=2000,
                  batch=64)
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
