"""Train a skill's state teacher with Brax PPO on the GPU.

  python skills/teacher.py --skill pick
  python skills/teacher.py --skill place --timesteps 200000000

Writes outputs/train/skill_<skill>_teacher/: params.pkl (normaliser, policy
and value parameters of the best evaluation), config.json (settings and control steps used) and
metrics.jsonl (one line per evaluation).
"""

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
# Run as a script: put the repository and DreamerV3 (for environments/) on
# the path, and let MuJoCo Warp allocate GPU memory beside JAX.
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402

TEACHER = dict(policy=(256, 256, 256), value=(256,) * 5)
# A learning rate of 1e-3 with 8 passes per batch made the pick teacher
# swing between 50% and 75% success and collapse late.
PPO = dict(
    episode_length=150, unroll_length=10, num_minibatches=32,
    num_updates_per_batch=4, discounting=0.97, learning_rate=3e-4,
    entropy_cost=1e-2, batch_size=192, num_evals=10, num_eval_envs=256,
    normalize_observations=True, reward_scaling=1.0,
    # Block velocities and absent-block slots are almost always constant;
    # without a floor on their running std, a rare change is divided by
    # ~1e-6 and the value loss blows up.
    normalize_observations_std_eps=0.05)
# Worlds per PPO batch; a multiple of batch_size * num_minibatches. 8192
# filled the 16 GB GPU once every step ended with a forward pass.
NUM_ENVS = 6144
# Starts the teacher trains on. Only about a fifth of pick targets sit on
# another block under the spec's mix, and those were the ones it failed;
# evaluation keeps the spec's mix.
TRAIN_STARTS = dict(pick=dict(prestack=0.6, stacked=0.5), place=dict())


def network_factory():
  from brax.training.agents.ppo import networks
  return functools.partial(
      networks.make_ppo_networks,
      policy_hidden_layer_sizes=TEACHER['policy'],
      value_hidden_layer_sizes=TEACHER['value'],
      policy_obs_key='state', value_obs_key='state')


def train(skill, out, num_envs=NUM_ENVS, timesteps=100_000_000, seed=0,
          **overrides):
  from brax.training.agents.ppo import train as ppo
  from skills.env import SkillEnv
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  settings = {**PPO, **overrides}
  log = (out / 'metrics.jsonl').open('w')
  last = {'step': 0}
  best = {'step': None, 'success': -1.0}
  start = time.time()

  def progress(step, metrics):
    row = {'step': int(step), 'seconds': time.time() - start}
    row.update({k: float(v) for k, v in metrics.items()})
    last.update(row)
    log.write(json.dumps(row) + '\n')
    log.flush()
    print(f"{step:>12,}  success {row.get('eval/episode_success', 0):.3f}"
          f"  reward {row.get('eval/episode_reward', 0):.2f}", flush=True)

  def keep_best(step, make_policy, params):
    # PPO here can collapse late, so the saved parameters are those of the
    # best evaluation so far rather than the last.
    del make_policy
    success = last.get('eval/episode_success', -1.0)
    if success > best['success']:
      best.update(step=int(step), success=success)
      with (out / 'params.pkl').open('wb') as f:
        pickle.dump(jax.device_get(params), f)

  ppo.train(
      environment=SkillEnv(skill, num_envs, **TRAIN_STARTS[skill]),
      eval_env=SkillEnv(skill, settings['num_eval_envs']),
      wrap_env=False, num_envs=num_envs, num_timesteps=timesteps,
      network_factory=network_factory(), seed=seed, progress_fn=progress,
      policy_params_fn=keep_best, **settings)
  config = dict(skill=skill, num_envs=num_envs, timesteps=timesteps,
                seed=seed, network=TEACHER, ppo=settings,
                control_steps=max(last['step'], timesteps),
                best_step=best['step'], best_success=best['success'],
                train_starts=TRAIN_STARTS[skill],
                seconds=time.time() - start)
  (out / 'config.json').write_text(json.dumps(config, indent=2))
  return config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--timesteps', type=int, default=100_000_000)
  parser.add_argument('--num-envs', type=int, default=NUM_ENVS)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--entropy-cost', type=float, default=PPO['entropy_cost'])
  args = parser.parse_args()
  out = REPO / 'outputs' / 'train' / f'skill_{args.skill}_teacher'
  config = train(args.skill, out, args.num_envs, args.timesteps, args.seed,
                 entropy_cost=args.entropy_cost)
  print(f"Wrote {out}: {config['control_steps']:,} control steps in "
        f"{config['seconds'] / 60:.0f} min")


if __name__ == '__main__':
  main()
