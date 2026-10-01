"""Evaluate a trained DreamerV3 agent on the real Atari emulator.

Plays --episodes full games (100 by default) and reports score and episode
length statistics. Several emulators run in lockstep so the agent steps them
as one batch on the GPU. Every game that starts is played to the end, so long
games are not dropped from the average.

  ~/anaconda3/envs/dreamer/bin/python planning/eval_model.py --logdir tetris
  ~/anaconda3/envs/dreamer/bin/python planning/eval_model.py --logdir tetris_400m_latest
  ~/anaconda3/envs/dreamer/bin/python planning/eval_model.py --logdir tetris_400m_latest \
      --planner gumbel --sims 32

By default the agent acts with its policy. The planners search inside the
world model instead, expanding nodes with the RSSM prior and scoring them with
the reward, continue and value heads:

  --planner gumbel  Gumbel MuZero search (mctx): the root draws up to
                    --considered actions from the policy without replacement
                    and Sequential Halving splits the --sims simulations
                    between them, dropping the worse half each round.
  --planner puct    MuZero's MCTS (mctx): every node, the root included,
                    descends by the PUCT rule with the policy as prior, and
                    the most visited root action is played. Root Dirichlet
                    noise is off, as it only serves exploration in training.
  --planner uct     UCT: UCB1 on scaled Q values with no policy prior, trying
                    every action once first. Leaves are valued by the mean
                    return of --rollouts policy rollouts of --horizon imagined
                    steps (plus the critic at their end with --bootstrap).

--logdir takes a path or the name of a run in outputs/train/. Emulator
settings (sticky actions, action set, image format, noops) come from the run's
config, so the agent is evaluated under the conditions it was trained in.
"""

import argparse
import json
import time

from world_model import build, eval_path, logdir_arg, make_env, seeder
from search import PUCT, make_search, uct_args, uct_options

import jax
import jax.numpy as jnp
import numpy as np


def evaluate(parts, envs, episodes, seed, plan=None):
  """Play `episodes` full games, one emulator per batch slot."""
  params, observe = parts['params'], parts['observe']
  B = len(envs)
  carry = parts['initial'](B)
  prevact = {'action': jax.device_put(np.zeros(B, np.int32))}
  acts = [{'action': np.int32(0), 'reset': True} for _ in range(B)]
  obs = [None] * B
  score, length = np.zeros(B), np.zeros(B, int)
  playing, started = np.ones(B, bool), B
  results, steps = [], 0
  while playing.any():
    for i in np.flatnonzero(playing):
      obs[i] = envs[i].step(acts[i])
    # Idle slots keep feeding their last frame; their actions are ignored.
    batch = {k: jax.device_put(np.stack([np.asarray(o[k]) for o in obs]))
             for k in parts['obs_space']}
    _, (carry, prevact) = observe(params, carry, batch, prevact, seed=seed())
    if plan:
      # The searched action replaces the policy sample, so it is also what the
      # RSSM sees as the previous action on the next step.
      prevact = {'action': plan(params, carry[1], seed())}
    actions = jax.device_get(prevact['action'])
    for i in np.flatnonzero(playing):
      o = obs[i]
      if o['is_first']:
        score[i], length[i] = 0.0, 0
      score[i] += float(o['reward'])
      length[i] += 1
      steps += 1
      acts[i] = {'action': actions[i], 'reset': bool(o['is_last'])}
      if not o['is_last']:
        continue
      results.append(dict(
          episode=len(results), score=float(score[i]), length=int(length[i]),
          truncated=not envs[i].ale.game_over()))
      r = results[-1]
      print(f'[{len(results):>{len(str(episodes))}}/{episodes}] '
            f'score {r["score"]:g}   length {r["length"]}'
            + ('   (hit the time limit)' if r['truncated'] else ''), flush=True)
      if started < episodes:
        started += 1
      else:
        playing[i] = False
  return results, steps


def histogram(values, width=40):
  """Text histogram: one row per value when there are few, else 10 bins."""
  distinct = np.unique(values)
  if len(distinct) <= 15:
    labels = [f'{v:g}' for v in distinct]
    counts = [int((values == v).sum()) for v in distinct]
  else:
    counts, edges = np.histogram(values, bins=10)
    labels = [f'{lo:g}–{hi:g}' for lo, hi in zip(edges[:-1], edges[1:])]
  pad = max(len(x) for x in labels)
  scale = width / max(counts)
  for label, count in zip(labels, counts):
    bar = '█' * int(round(count * scale)) if count else ''
    print(f'  {label:>{pad}} │ {bar} {count}')


def report(results, logdir, elapsed, steps, limit):
  scores = np.array([r['score'] for r in results])
  lengths = np.array([r['length'] for r in results])
  n = len(scores)
  sem = scores.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
  q25, med, q75 = np.percentile(scores, [25, 50, 75])
  truncated = sum(r['truncated'] for r in results)

  print(f'\n{n} episodes in {elapsed / 60:.1f} min '
        f'({steps / elapsed:.0f} env steps/s)\n')
  print(f'Score    mean {scores.mean():.2f} ± {1.96 * sem:.2f} (95% CI)   '
        f'std {scores.std(ddof=1) if n > 1 else 0.0:.2f}')
  print(f'         median {med:g}   IQR {q25:g}–{q75:g}   '
        f'min {scores.min():g}   max {scores.max():g}')
  print(f'         {(scores > 0).mean():.0%} of games scored above zero')
  print(f'Length   mean {lengths.mean():.0f}   median {np.median(lengths):.0f}   '
        f'min {lengths.min()}   max {lengths.max()}   '
        f'({truncated} hit the {limit} limit)')
  print('\nScore distribution')
  histogram(scores)

  path = logdir / 'scores.jsonl'
  if path.exists():
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    train = np.array([r['episode/score'] for r in rows[-n:]])
    if len(train):
      print(f'\nTraining reference: last {len(train)} episodes in scores.jsonl '
            f'averaged {train.mean():.2f} (up to step {rows[-1]["step"]})')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=logdir_arg, default='tetris',
                      help='run directory, or a run name in outputs/train/')
  parser.add_argument('--episodes', type=int, default=100)
  parser.add_argument('--envs', type=int, default=16,
                      help='emulators stepped in parallel as one batch')
  parser.add_argument('--max-steps', type=int, default=None,
                      help='agent steps before a game is cut off '
                           '(default: the emulator limit of 108000 frames)')
  parser.add_argument('--planner', choices=['policy', 'gumbel', 'puct', 'uct'],
                      default='policy',
                      help='act with the policy, Gumbel MuZero search, '
                           'MCTS with PUCT, or UCT with policy rollouts')
  parser.add_argument('--sims', type=int, default=32,
                      help='search simulations per step (planners only)')
  parser.add_argument('--considered', type=int, default=16,
                      help='root actions Sequential Halving starts from '
                           '(--planner gumbel; capped at the action count)')
  uct_args(parser)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', default=None,
                      help='JSONL file for the per-episode results; a bare file '
                           'name goes in outputs/eval/<run>/ (default: '
                           '<planner>_<time>.jsonl there)')
  args = parser.parse_args()
  rng, seed = seeder(args.seed)

  print('Loading the agent...')
  parts = build(args.logdir)
  parts['env'].close()
  config = parts['config']
  # Custom environments' actions take a varying number of frames, so their
  # limit stays in frames.
  suite = config.task.split('_', 1)[0]
  repeat = None if suite == 'custom' else config.env[suite]['repeat']
  if args.max_steps is not None and repeat is None:
    parser.error('--max-steps needs an environment with a fixed frame repeat')
  overrides = {} if args.max_steps is None else {
      'length': args.max_steps * repeat}
  envs = [make_env(config, i, seed=int(rng.integers(0, 2 ** 31)), **overrides)
          for i in range(min(args.envs, args.episodes))]
  limit = (f'{envs[0].length // repeat}-step' if repeat else
           f'{envs[0].length}-frame')
  print(f'{parts["task"]} ({args.logdir.name}, checkpoint {parts["ckpt"]}): '
        f'{args.episodes} episodes on {len(envs)} emulators')

  plan = None
  if args.planner == 'gumbel':
    considered = min(args.considered, len(parts['actions']))
    search = make_search(parts['model'], args.sims, 'gumbel',
                         max_num_considered_actions=considered)
    print(f'Planning with Gumbel MuZero search: {args.sims} simulations, '
          f'Sequential Halving over {considered} of '
          f'{len(parts["actions"])} actions')
  elif args.planner == 'puct':
    search = make_search(parts['model'], args.sims, 'puct', **PUCT)
    print(f'Planning with MCTS (PUCT, c1 {PUCT["pb_c_init"]:g}, '
          f'c2 {PUCT["pb_c_base"]:g}): {args.sims} simulations, playing the '
          f'most visited action')
  elif args.planner == 'uct':
    options = uct_options(args)
    search = make_search(parts['model'], args.sims, 'uct', **options)
    print(f'Planning with UCT: {args.sims} simulations, c {args.c:.3g}, '
          f'{args.rollouts} policy rollouts of {args.horizon} steps per leaf'
          + (' plus the critic' if args.bootstrap else '')
          + ', playing the most visited action')
  else:
    print('Acting with the policy')
  if args.planner != 'policy':
    plan = jax.jit(lambda *a: search(*a).action.astype(jnp.int32))

  start = time.time()
  results, steps = evaluate(parts, envs, args.episodes, seed, plan)
  elapsed = time.time() - start
  for env in envs:
    env.close()

  out = eval_path(args.logdir, args.out or
                  f'{args.planner}_{time.strftime("%Y%m%dT%H%M%S")}.jsonl')
  with open(out, 'w') as f:
    for r in results:
      f.write(json.dumps(r) + '\n')
  report(results, args.logdir, elapsed, steps, limit)
  print(f'\nPer-episode results written to {out}')


if __name__ == '__main__':
  main()
