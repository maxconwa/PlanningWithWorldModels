"""Play an Atari game inside a trained DreamerV3 world model.

The real emulator only supplies the first frame of a fresh episode to seed the
RSSM state (raise --warmup to start from later in a game). After that the
emulator is closed and every frame you see is hallucinated by the model from
your own keystrokes.

  ~/anaconda3/envs/dreamer/bin/python planning/dream_atari.py --logdir tetris
  ~/anaconda3/envs/dreamer/bin/python planning/dream_atari.py --logdir breakout_69k
  ~/anaconda3/envs/dreamer/bin/python planning/dream_atari.py --logdir tetris_400m_latest

--logdir takes a path or the name of a run in outputs/train/. Image format,
action set and model size come from the run's config, so full-action grayscale
runs, minimal-action color runs and the 400M-parameter Tetris run all work.

Controls: arrows + space (FIRE) play, TAB hands over to the trained policy,
R rewinds to the seed frame, ESC quits.
"""

import argparse

from world_model import build, logdir_arg, seeder, warmup

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pygame

# ALE action meanings for the held keys, in the order we check them. Keys whose
# action is not in the run's action set are ignored.
KEYS = {
    pygame.K_LEFT: 'LEFT', pygame.K_RIGHT: 'RIGHT', pygame.K_UP: 'UP',
    pygame.K_SPACE: 'FIRE', pygame.K_DOWN: 'DOWN'}


def make_dream(model):
  """Jitted imagined step for the model returned by world_model.build."""
  from dreamerv3.agent import sample

  def dream(dyn_carry, action):
    """One imagined step: no frame goes in, the model predicts the next one."""
    dyn_carry, (feat, _) = model.dyn.imagine(
        dyn_carry, action, 1, False, single=True)
    _, _, recons = model.dec(
        {}, feat, jnp.zeros(len(feat['deter']), bool), False, single=True)
    inp = model.feat2tensor(feat)
    return dyn_carry, dict(
        image=jnp.clip(255 * recons['image'].pred(), 0, 255).astype(jnp.uint8),
        reward=model.rew(inp, 1).pred(),
        cont=model.con(inp, 1).prob(1),
        policy=sample(model.pol(inp, 1)))

  return jax.jit(nj.pure(dream))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=logdir_arg, default='tetris',
                      help='run directory, or a run name in outputs/train/')
  parser.add_argument('--warmup', type=int, default=1,
                      help='real emulator frames observed before dreaming')
  parser.add_argument('--scale', type=int, default=None,
                      help='pixels per frame pixel (default: ~576px wide window)')
  parser.add_argument('--fps', type=int, default=8)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  _, seed = seeder(args.seed)

  print('Loading the world model...')
  parts = build(args.logdir)
  actions = parts['actions']
  keys = {k: actions.index(m) for k, m in KEYS.items() if m in actions}
  print(f'{parts["task"]}: {len(actions)} actions {actions}')
  print(f'Seeding the dream with {args.warmup} real frame(s)...')
  start, _ = warmup(parts, args.warmup, seed)
  params, dream = parts['params'], make_dream(parts['model'])

  H, W, _ = parts['obs_space']['image'].shape
  scale = args.scale or max(1, 576 // W)
  pygame.init()
  screen = pygame.display.set_mode((W * scale, H * scale + 36))
  pygame.display.set_caption(
      f'{parts["task"]} in the Dreamer world model ({args.logdir.name})')
  font = pygame.font.Font(None, 24)
  clock = pygame.time.Clock()

  carry, total, steps, auto, polact = start, 0.0, 0, False, 0
  running = True
  while running:
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        running = False
      elif event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
          running = False
        elif event.key == pygame.K_r:
          carry, total, steps = start, 0.0, 0
        elif event.key == pygame.K_TAB:
          auto = not auto

    held = pygame.key.get_pressed()
    action = polact if auto else next((v for k, v in keys.items() if held[k]), 0)
    act = {'action': jax.device_put(np.array([action], np.int32))}
    _, (carry, out) = dream(params, carry, act, seed=seed())
    out = jax.device_get(out)
    polact = int(out['policy']['action'][0])
    total += float(out['reward'][0])
    steps += 1

    frame = out['image'][0]
    if frame.shape[-1] == 1:
      frame = np.repeat(frame, 3, -1)
    surf = pygame.transform.scale(
        pygame.surfarray.make_surface(frame.transpose(1, 0, 2)),
        (W * scale, H * scale))
    screen.fill((16, 16, 16))
    screen.blit(surf, (0, 0))
    hud = (f'step {steps}   dreamed return {total:+.1f}   '
           f'cont {float(out["cont"][0]):.2f}' + ('   AUTOPILOT' if auto else ''))
    screen.blit(font.render(hud, True, (210, 210, 210)), (8, H * scale + 10))
    pygame.display.flip()
    clock.tick(args.fps)

  pygame.quit()


if __name__ == '__main__':
  main()
