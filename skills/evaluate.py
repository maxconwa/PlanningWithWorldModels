"""Success rate of a skill policy over fresh starts: the gates of stage 2.

  python skills/evaluate.py --skill pick --who teacher
  python skills/evaluate.py --skill place --who student --episodes 500

Plays one episode in each of --episodes worlds and writes
outputs/eval/skill_<skill>/<who>_gate.json.
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
import json  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

def success_rate(skill, policy, episodes=500, cameras=False, seed=0):
  """Share of worlds whose first episode ends in success."""
  from skills.env import SkillEnv
  from skills.tasks import EPISODE
  env = SkillEnv(skill, episodes, cameras=cameras)
  keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
  state = jax.jit(env.reset)(keys)
  step = jax.jit(lambda s: env.step(s, policy(s.obs)))
  ended = np.zeros(episodes, bool)
  success = np.zeros(episodes, bool)
  for _ in range(EPISODE):
    state = step(state)
    done = np.asarray(state.done) > 0
    first = done & ~ended
    success[first] = np.asarray(state.metrics['success'])[first] > 0
    ended |= done
    if ended.all():
      break
  return float(success.mean())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--who', choices=['teacher', 'student', 'scripted'],
                      required=True)
  parser.add_argument('--episodes', type=int, default=500)
  parser.add_argument('--seed', type=int, default=1)
  args = parser.parse_args()
  from skills import policy as policies
  from skills import scripted
  run = REPO / 'outputs' / 'train' / f'skill_{args.skill}_{args.who}'
  if args.who == 'teacher':
    act = policies.load_teacher(run)
    policy, cameras = (lambda obs: act(obs['state'])), False
  elif args.who == 'student':
    policy, cameras = policies.load_student(run), True
  else:
    script = getattr(scripted, args.skill)
    policy, cameras = (lambda obs: script(obs['state'])), False
  rate = success_rate(args.skill, policy, args.episodes, cameras, args.seed)
  out = REPO / 'outputs' / 'eval' / f'skill_{args.skill}'
  out.mkdir(parents=True, exist_ok=True)
  result = dict(skill=args.skill, who=args.who, episodes=args.episodes,
                seed=args.seed, success=rate, passed=rate >= 0.9)
  (out / f'{args.who}_gate.json').write_text(json.dumps(result, indent=2))
  print(json.dumps(result))


if __name__ == '__main__':
  main()
