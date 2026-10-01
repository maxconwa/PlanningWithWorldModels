"""Watch a trained agent play the real environment in a window.

  python planning/watch.py --logdir tetris_placement_12m
  python planning/watch.py --logdir tetris_placement_12m --planner puct --sims 64
  python planning/watch.py --logdir tetris_placement_12m --planner policy

Before each action the agent searches its world model from its current state
(--planner gumbel, Gumbel MuZero, by default; or puct, MuZero's MCTS) with
leaves valued by the critic, and plays the action the search picks. With
--planner policy it acts with its policy alone. The window shows the
observation the agent gets, scaled up, and the status line shows the action
played next to the one the policy would have played.
Keys: Space pauses, S steps once while paused, + and - change the speed,
N starts a new game and Escape quits. Every step is also printed to the
terminal.
"""

import argparse
import tkinter

from world_model import build, logdir_arg, seeder
from search import PUCT, make_search

import jax
import jax.numpy as jnp
import numpy as np
from PIL import ImageTk

from environments.play import render


class Watcher:

  def __init__(self, parts, plan, seed, fps, scale):
    self.parts, self.plan, self.seed, self.scale = parts, plan, seed, scale
    self.env, self.names = parts['env'], parts['actions']
    self.delay = int(1000 / fps)
    self.paused, self.games, self.scores = False, 0, []
    self.root = tkinter.Tk()
    self.root.title(f'{parts["task"]}: agent from checkpoint {parts["ckpt"]}'
                    + (' with tree search' if plan else ''))
    self.view = tkinter.Label(self.root, bg='black')
    self.view.pack()
    self.status = tkinter.Label(
        self.root, font='TkFixedFont', anchor='w', justify='left')
    self.status.pack(fill='x', padx=8, pady=4)
    tkinter.Label(
        self.root, anchor='w', fg='gray40',
        text='Space pause, S step, +/- speed, N new game, Esc quit').pack(
            fill='x', padx=8, pady=(0, 6))
    self.root.bind('<Key>', self.press)
    self.reset()
    self.root.after(self.delay, self.tick)

  def reset(self):
    """Start a new game; the agent's state starts fresh with it."""
    self.carry = self.parts['initial']()
    self.prevact = {'action': jax.device_put(np.zeros(1, np.int32))}
    self.score, self.steps, self.games = 0.0, 0, self.games + 1
    obs = self.env.step({'reset': True, 'action': np.int32(0)})
    self.observe(obs, 'new game')

  def observe(self, obs, what):
    """Show `obs` and let the agent pick its next action from it."""
    self.obs = obs
    batch = {k: jax.device_put(np.asarray(obs[k])[None])
             for k in self.parts['obs_space']}
    _, (self.carry, self.prevact) = self.parts['observe'](
        self.parts['params'], self.carry, batch, self.prevact,
        seed=self.seed())
    self.policy = int(jax.device_get(self.prevact['action'])[0])
    if self.plan:
      # The searched action replaces the policy sample, so it is also what
      # the RSSM sees as the previous action on the next step.
      self.prevact = {'action': self.plan(
          self.parts['params'], self.carry[1], self.seed())}
    self.action = int(jax.device_get(self.prevact['action'])[0])
    self.photo = ImageTk.PhotoImage(
        render(self.env, obs, self.scale, screen=False))
    self.view.configure(image=self.photo)
    mean = f'{np.mean(self.scores):.1f}' if self.scores else '-'
    line = (f'game {self.games:<3} step {self.steps:<4} {what:<12} '
            f'reward {float(obs["reward"]):+g}  score {self.score:g}  '
            f'(previous games: mean {mean}, best '
            f'{max(self.scores, default=0):g})')
    if self.plan and not obs['is_last']:
      line += (f'\nnext: {self.names[self.action]} by search, '
               f'{self.names[self.policy]} by the policy')
    if obs['is_last']:
      line += '  GAME OVER'
    if self.paused:
      line += '  PAUSED'
    self.status.configure(text=line)
    print(line, flush=True)

  def advance(self):
    if self.obs['is_last']:
      self.scores.append(self.score)
      self.reset()
      return
    obs = self.env.step({'reset': False, 'action': np.int32(self.action)})
    self.score += float(obs['reward'])
    self.steps += 1
    self.observe(obs, self.names[self.action])

  def tick(self):
    if not self.paused:
      self.advance()
    self.root.after(self.delay, self.tick)

  def press(self, event):
    key = event.keysym
    if key == 'Escape':
      self.root.destroy()
    elif key == 'space':
      self.paused = not self.paused
    elif key in ('s', 'S') and self.paused:
      self.advance()
    elif key in ('plus', 'equal', 'KP_Add'):
      self.delay = max(10, self.delay // 2)
    elif key in ('minus', 'KP_Subtract'):
      self.delay = min(4000, self.delay * 2)
    elif key in ('n', 'N'):
      self.reset()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=logdir_arg, required=True,
                      help='run directory, or a run name in outputs/train/')
  parser.add_argument('--planner', choices=['gumbel', 'puct', 'policy'],
                      default='gumbel',
                      help='Gumbel MuZero search, MCTS with PUCT, or the '
                           'policy alone')
  parser.add_argument('--sims', type=int, default=32,
                      help='search simulations per action')
  parser.add_argument('--considered', type=int, default=16,
                      help='root actions Sequential Halving starts from '
                           '(--planner gumbel; capped at the action count)')
  parser.add_argument('--fps', type=float, default=4,
                      help='agent actions per second')
  parser.add_argument('--scale', type=int, default=6,
                      help='pixels per observation pixel')
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  _, seed = seeder(args.seed)
  print('Loading the agent...')
  parts = build(args.logdir, env_seed=args.seed)
  print(f'{parts["task"]}: {len(parts["actions"])} actions {parts["actions"]}')
  plan = None
  if args.planner == 'gumbel':
    considered = min(args.considered, len(parts['actions']))
    search = make_search(parts['model'], args.sims, 'gumbel',
                         max_num_considered_actions=considered)
    print(f'Planning with Gumbel MuZero: {args.sims} simulations over '
          f'{considered} actions')
  elif args.planner == 'puct':
    search = make_search(parts['model'], args.sims, 'puct', **PUCT)
    print(f'Planning with MCTS (PUCT): {args.sims} simulations')
  if args.planner != 'policy':
    plan = jax.jit(lambda *a: search(*a).action.astype(jnp.int32))
  Watcher(parts, plan, seed, args.fps, args.scale).root.mainloop()


if __name__ == '__main__':
  main()
