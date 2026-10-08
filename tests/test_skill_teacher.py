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
  assert a.shape == (5, 5) and np.abs(a).max() <= 1.0
  np.testing.assert_allclose(a, b, atol=1e-6)
  again = np.asarray(load_teacher(tmp_path)(obs))
  np.testing.assert_allclose(a, again, atol=1e-6)


def test_teacher_keeps_its_best_evaluation(tmp_path):
  """PPO can collapse late; the saved parameters are the best evaluated."""
  config = train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 8,
                 batch_size=2, num_minibatches=32, num_evals=4,
                 num_eval_envs=16)
  rows = [json.loads(line) for line in
          (tmp_path / 'metrics.jsonl').read_text().splitlines()]
  best = max(rows, key=lambda r: r['eval/episode_success'])
  assert config['best_step'] == best['step']
  assert config['best_success'] == best['eval/episode_success']


def test_teacher_trains_on_more_stacked_pick_starts(tmp_path):
  """Training favours stacked pick targets; evaluation keeps the spec's
  start mix, so the gate measures the real task."""
  from skills.teacher import PPO, TRAIN_STARTS
  assert TRAIN_STARTS['pick']['stacked'] > 0
  assert PPO['learning_rate'] <= 3e-4
  config = train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4,
                 batch_size=2, num_minibatches=32, num_evals=2,
                 num_eval_envs=16)
  assert config['train_starts'] == TRAIN_STARTS['pick']


def test_observation_normaliser_has_a_floor():
  """Block velocities and absent-block slots are almost always constant; a
  zero floor on their running std divided rare changes by ~1e-6 and blew the
  value loss up to 1e11."""
  from skills.teacher import NUM_ENVS, PPO
  assert PPO['normalize_observations_std_eps'] >= 0.01
  assert NUM_ENVS % (PPO['batch_size'] * PPO['num_minibatches']) == 0


def test_best_keeper_saves_the_parameters_that_were_evaluated():
  """Brax hands over parameters before evaluating them (after, at step 0);
  the kept parameters must be the ones whose evaluation was best."""
  from skills.teacher import BestKeeper
  saved = []
  keeper = BestKeeper(lambda step, params: saved.append((step, params)))
  keeper.evaluated(0, 0.0)
  keeper.received(0, 'p0')
  for step, success in ((10, 0.5), (20, 0.8), (30, 0.6)):
    keeper.received(step, f'p{step}')
    keeper.evaluated(step, success)
  assert saved[-1] == (20, 'p20')
  assert (keeper.step, keeper.success) == (20, 0.8)


def test_saved_teacher_matches_the_trained_one(tmp_path):
  """load_teacher rebuilds exactly what Brax trained (network and
  normaliser), from the best-evaluated parameters."""
  config, make_policy, params = train(
      'pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4, batch_size=2,
      num_minibatches=32, num_evals=2, num_eval_envs=16, return_policy=True)
  obs = jax.random.normal(jax.random.PRNGKey(2), (6, 85))
  brax = make_policy(params, deterministic=True)(
      {'state': obs}, jax.random.PRNGKey(0))[0]
  np.testing.assert_allclose(np.asarray(load_teacher(tmp_path)(obs)),
                             np.asarray(brax), atol=1e-6)


def test_short_evaluations_do_not_overwrite_the_gate():
  from skills.evaluate import result_path
  assert result_path('pick', 'student', 500).name == 'student_gate.json'
  assert result_path('pick', 'student', 100).name == 'student_100.json'
