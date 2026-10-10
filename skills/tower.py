"""Build towers through the option environment, choosing options from the
simulator state: the check that the trained skills chain into stacks.

  python skills/tower.py --n 3
  python skills/tower.py --n 2 --episodes 128

Plays one episode in each of --episodes worlds and writes
outputs/eval/stack_options/scripted_n<n>.json.
"""

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

COLOURS = 4
TABLE = 2 * COLOURS
NEAR_XY = 0.03


def top_of(blocks, c):
  """The highest present block in c's stack, at or above c."""
  pos = blocks[:, 1:4]
  stack = ((blocks[:, 0] > 0) &
           (np.linalg.norm(pos[:, :2] - pos[c, :2], axis=-1) < NEAR_XY) &
           (pos[:, 2] >= pos[c, 2]))
  return int(np.argmax(np.where(stack, pos[:, 2], -1.0)))


def choose(blocks, held, mask, goal, n, k):
  """The next option for one world: clear the tower top and the next goal
  block, then pick the next block and put it on the tower."""
  if k >= n:
    return int(np.flatnonzero(~mask)[0])  # Hold still: any invalid option.
  want = int(goal[k])
  if held >= 0:
    if held == want:
      return TABLE if k == 0 else COLOURS + int(goal[k - 1])
    return TABLE
  for c in ([int(goal[k - 1])] if k > 0 else []) + [want]:
    if not mask[c]:
      return top_of(blocks, c)
  return want


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=3, help='blocks per tower')
  parser.add_argument('--episodes', type=int, default=64)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--max-options', type=int, default=60,
                      help='decisions before an episode counts as failed')
  args = parser.parse_args()
  from environments.stack_options import BatchStackOptions
  env = BatchStackOptions(args.episodes, n=args.n, seed=args.seed)
  b = args.episodes
  obs = env.step({'reset': np.ones(b, bool), 'action': np.zeros(b, np.int32)})
  ended = np.zeros(b, bool)
  success = np.zeros(b, bool)
  options = np.zeros(b, int)
  steps = np.zeros(b, int)
  durations = []
  for _ in range(args.max_options):
    priv = env.privileged()
    action = np.array([choose(
        priv['blocks'][i], priv['held'][i], priv['option_mask'][i],
        obs['goal'][i], obs['log/n'][i], obs['log/prefix'][i])
        for i in range(b)], np.int32)
    valid = priv['option_mask'][np.arange(b), action]
    obs = env.step({'reset': np.zeros(b, bool), 'action': action})
    live = ~ended
    options += live
    steps += np.where(live, obs['log/duration'], 0)
    durations += list(obs['log/duration'][live & valid])
    success |= live & obs['is_terminal']
    ended |= obs['is_last']
    if ended.all():
      break
  result = dict(
      n=args.n, episodes=b, seed=args.seed,
      success=float(success.mean()),
      option_steps=float(np.mean(durations)),
      options=float(options.mean()), control_steps=float(steps.mean()))
  path = REPO / 'outputs' / 'eval' / 'stack_options' / f'scripted_n{args.n}.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2))
  print(json.dumps(result))


if __name__ == '__main__':
  main()
