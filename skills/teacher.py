"""Train a skill's state teacher with Brax PPO on the GPU.

  python skills/teacher.py --skill pick
  python skills/teacher.py --skill place --timesteps 200000000

Writes outputs/train/skill_<skill>_teacher/: params.pkl (normaliser, policy
and value parameters), config.json (settings and control steps used) and
metrics.jsonl (one line per evaluation).
"""

import argparse
import functools
import json
import pathlib
import pickle
import time

import jax

REPO = pathlib.Path(__file__).resolve().parent.parent

TEACHER = dict(policy=(256, 256, 256), value=(256,) * 5)
PPO = dict(
    episode_length=150, unroll_length=10, num_minibatches=32,
    num_updates_per_batch=8, discounting=0.97, learning_rate=1e-3,
    entropy_cost=1e-2, batch_size=256, num_evals=10, num_eval_envs=256,
    normalize_observations=True, reward_scaling=1.0)


def network_factory():
  from brax.training.agents.ppo import networks
  return functools.partial(
      networks.make_ppo_networks,
      policy_hidden_layer_sizes=TEACHER['policy'],
      value_hidden_layer_sizes=TEACHER['value'],
      policy_obs_key='state', value_obs_key='state')


def train(skill, out, num_envs=8192, timesteps=100_000_000, seed=0,
          **overrides):
  from brax.training.agents.ppo import train as ppo
  from skills.env import SkillEnv
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  settings = {**PPO, **overrides}
  log = (out / 'metrics.jsonl').open('w')
  last = {'step': 0}
  start = time.time()

  def progress(step, metrics):
    row = {'step': int(step), 'seconds': time.time() - start}
    row.update({k: float(v) for k, v in metrics.items()})
    last.update(row)
    log.write(json.dumps(row) + '\n')
    log.flush()
    print(f"{step:>12,}  success {row.get('eval/episode_success', 0):.3f}"
          f"  reward {row.get('eval/episode_reward', 0):.2f}", flush=True)

  _, params, _ = ppo.train(
      environment=SkillEnv(skill, num_envs),
      eval_env=SkillEnv(skill, settings['num_eval_envs']),
      wrap_env=False, num_envs=num_envs, num_timesteps=timesteps,
      network_factory=network_factory(), seed=seed, progress_fn=progress,
      **settings)
  with (out / 'params.pkl').open('wb') as f:
    pickle.dump(jax.device_get(params), f)
  config = dict(skill=skill, num_envs=num_envs, timesteps=timesteps,
                seed=seed, network=TEACHER, ppo=settings,
                control_steps=max(last['step'], timesteps),
                seconds=time.time() - start)
  (out / 'config.json').write_text(json.dumps(config, indent=2))
  return config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--timesteps', type=int, default=100_000_000)
  parser.add_argument('--num-envs', type=int, default=8192)
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
