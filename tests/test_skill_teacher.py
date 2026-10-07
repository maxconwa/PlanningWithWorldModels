import json

import jax
import numpy as np

from skills import scripted
from skills.evaluate import success_rate
from skills.policy import load_teacher
from skills.teacher import train


def test_scripted_policy_passes_through_the_evaluator():
  # The script succeeds in about 84% of 500 starts; 64 starts vary by +-5%.
  rate = success_rate('pick', lambda obs: scripted.pick(obs['state']),
                      episodes=64)
  assert rate >= 0.7


def test_short_teacher_run_saves_a_policy(tmp_path):
  config = train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4,
                 batch_size=2, num_minibatches=32, num_evals=2,
                 num_eval_envs=16)
  assert (tmp_path / 'params.pkl').exists()
  assert json.loads((tmp_path / 'config.json').read_text())['skill'] == 'pick'
  assert config['control_steps'] >= 64 * 10 * 4
  assert (tmp_path / 'metrics.jsonl').read_text().strip()


def test_saved_teacher_reloads(tmp_path):
  train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4, batch_size=2,
        num_minibatches=32, num_evals=2, num_eval_envs=16)
  policy = load_teacher(tmp_path)
  obs = jax.random.normal(jax.random.PRNGKey(0), (5, 85))
  a, b = np.asarray(policy(obs)), np.asarray(jax.jit(policy)(obs))
  assert a.shape == (5, 6) and np.abs(a).max() <= 1.0
  np.testing.assert_allclose(a, b, atol=1e-6)
  again = np.asarray(load_teacher(tmp_path)(obs))
  np.testing.assert_allclose(a, again, atol=1e-6)
