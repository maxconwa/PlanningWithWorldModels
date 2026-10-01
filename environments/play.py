"""Play a custom environment by hand, to check it before training on it.

  python environments/play.py tetris
  python environments/play.py tetris --option gray=False --option sticky=False
  python environments/play.py catch

The window shows the observation the agent gets, scaled up, and for Atari
environments the full emulator screen next to it. Options default to the ones
training uses (DreamerV3's config section named by the environment's
OPTIONS), and --option key=value overrides one. Keys are listed in KEYS and
shown in the window; N starts a new episode and Escape quits. Every step is
also printed to the terminal.
"""

import argparse
import pathlib
import sys
import tkinter

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]

import numpy as np  # noqa: E402
import ruamel.yaml as yaml  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402

import environments  # noqa: E402

CONFIGS = REPO / 'third_party' / 'dreamerv3' / 'dreamerv3' / 'configs.yaml'

# Tk key names to actions, per environment.
KEYS = {
    'catch': {'Left': 0, 'Down': 1, 'Right': 2},
    'tetris': {**{str(c): c for c in range(10)}, 'r': 10, 'Up': 10},
}
HELP = {
    'catch': 'Left/Right move the paddle, Down keeps it still',
    'tetris': '0-9 place the piece with its leftmost cell in that column, '
              'R or Up rotates',
}


def options(name, overrides):
  """The environment's training options, with key=value overrides applied."""
  section = getattr(environments.lookup(name), 'OPTIONS', 'custom')
  config = yaml.YAML(typ='safe').load(CONFIGS.read_text())
  opts = dict(config['defaults']['env'].get(section, {}))
  for item in overrides:
    key, value = item.split('=', 1)
    opts[key] = yaml.YAML(typ='safe').load(value)
  if 'size' in opts:
    opts['size'] = tuple(opts['size'])
  return opts


def render(env, obs, scale, screen=True):
  """The observation, after the emulator screen for Atari environments when
  `screen` is set."""
  image = obs['image']
  if image.shape[-1] == 1:
    image = np.repeat(image, 3, -1)
  panels = [Image.fromarray(image).resize(
      (image.shape[1] * scale, image.shape[0] * scale), Image.NEAREST)]
  if screen:
    try:
      screen = Image.fromarray(env.ale.getScreenRGB())
    except (AttributeError, ValueError):  # DreamerV3's wrappers raise these.
      screen = None
  if screen:
    panels.insert(0, screen.resize(
        (screen.width * 2, screen.height * 2), Image.NEAREST))
  height = max(p.height for p in panels)
  canvas = Image.new(
      'RGB', (sum(p.width for p in panels) + 8 * (len(panels) - 1), height))
  x = 0
  for panel in panels:
    canvas.paste(panel, (x, (height - panel.height) // 2))
    x += panel.width + 8
  return canvas


class Player:

  def __init__(self, name, env, scale):
    self.name, self.env, self.scale = name, env, scale
    self.keys = KEYS.get(name, {str(i): i for i in range(10)})
    self.root = tkinter.Tk()
    self.root.title(f'{name}: manual play')
    self.view = tkinter.Label(self.root, bg='black')
    self.view.pack()
    self.status = tkinter.Label(
        self.root, font='TkFixedFont', anchor='w', justify='left')
    self.status.pack(fill='x', padx=8, pady=4)
    hint = HELP.get(name, 'digits pick the action')
    hint += '; N new episode, Esc quit'
    tkinter.Label(self.root, text=hint, anchor='w', fg='gray40').pack(
        fill='x', padx=8, pady=(0, 6))
    self.root.bind('<Key>', self.press)
    self.reset()

  def reset(self):
    self.score, self.steps = 0.0, 0
    obs = self.env.step({'reset': True, 'action': 0})
    self.show(obs, 'new episode')

  def press(self, event):
    if event.keysym == 'Escape':
      self.root.destroy()
    elif event.keysym in ('n', 'N'):
      self.reset()
    elif event.keysym in self.keys and not self.done:
      action = self.keys[event.keysym]
      obs = self.env.step({'reset': False, 'action': action})
      self.score += float(obs['reward'])
      self.steps += 1
      self.show(obs, f'action {action} ({event.keysym})')

  def show(self, obs, what):
    self.done = bool(obs['is_last'])
    self.photo = ImageTk.PhotoImage(render(self.env, obs, self.scale))
    self.view.configure(image=self.photo)
    line = (f'step {self.steps:<4} {what:<18} reward {float(obs["reward"]):+g}'
            f'   score {self.score:g}')
    if hasattr(self.env, 'duration'):
      line += f'   frames {self.env.duration}'
    if self.done:
      line += '   EPISODE OVER (press N)'
    self.status.configure(text=line)
    print(line, flush=True)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('name', choices=sorted(environments.REGISTRY),
                      help='registered environment')
  parser.add_argument('--option', action='append', default=[],
                      help='key=value environment option, e.g. gray=False')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--scale', type=int, default=4,
                      help='pixels per observation pixel')
  args = parser.parse_args()
  opts = options(args.name, args.option)
  print(f'{args.name} options: {opts}')
  env = environments.make(args.name, seed=args.seed, **opts)
  Player(args.name, env, args.scale).root.mainloop()


if __name__ == '__main__':
  main()
