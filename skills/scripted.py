"""Scripted pick and place from privileged state, used to check that each
skill environment can be solved and that its success is detected. They read
only obs['state'] (layout in skills/env.py)."""

import jax.numpy as jnp
import numpy as np

from environments.stack_control import MAX_STEP, MAX_YAW
from environments.stack_scene import HALF

ALIGN = 0.008  # How close in xy before descending, metres.
YAW_OK = 0.05
# Actions move the controller's target, which the arm follows with a lag;
# commanding the full error each step would overshoot, so close a share.
GAIN = 0.3


def _split(state):
  b = state.shape[0]
  blocks = state[:, 19:75].reshape(b, 4, 14)
  return dict(
      ee=state[:, 14:17], yaw=state[:, 17], opening=state[:, 18],
      present=blocks[..., 0] > 0, pos=blocks[..., 4:7],
      quat=blocks[..., 7:11], target=jnp.argmax(state[:, 75:80], -1),
      held=state[:, 80:84] > 0)


def _block_yaw(quat):
  w, x, y, z = (quat[..., i] for i in range(4))
  return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _turn(ee_yaw, block_yaw):
  """Yaw step that lines the fingers up with the nearest pair of faces."""
  err = (block_yaw - ee_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
  return jnp.clip(GAIN * err / MAX_YAW, -1, 1), jnp.abs(err) < YAW_OK


def _act(ee, want, turn, grip):
  move = jnp.clip(GAIN * (want - ee) / MAX_STEP, -1, 1)
  return jnp.concatenate([move, turn[:, None], grip[:, None]], -1)


def pick(state):
  s = _split(state)
  rows = jnp.arange(state.shape[0])
  p = s['pos'][rows, s['target']]
  held = s['held'][rows, s['target']]
  turn, aligned = _turn(s['yaw'], _block_yaw(s['quat'][rows, s['target']]))
  xy_ok = jnp.linalg.norm(p[:, :2] - s['ee'][:, :2], axis=-1) < ALIGN
  above = p + jnp.array([0.0, 0.0, 0.08])
  want = jnp.where((xy_ok & aligned)[:, None], p, above)
  near = jnp.linalg.norm(s['ee'] - p, axis=-1) < ALIGN
  want = jnp.where(held[:, None], s['ee'] + jnp.array([0.0, 0.0, 0.03]),
                   want)
  grip = jnp.where(held | near, 1.0, -1.0)
  return _act(s['ee'], want, turn, grip)


def place(state):
  s = _split(state)
  rows = jnp.arange(state.shape[0])
  holding = s['held'].any(-1)
  nearest = jnp.argmin(jnp.where(
      s['present'], jnp.linalg.norm(
          s['pos'][..., :2] - s['ee'][:, None, :2], axis=-1), 9.0), -1)
  h = jnp.where(holding, jnp.argmax(s['held'], -1), nearest)
  p = s['pos'][rows, h]
  t = s['target']
  on_block = s['pos'][rows, jnp.minimum(t, 3)] + jnp.array([0, 0, 2 * HALF])
  table = jnp.stack([p[:, 0], p[:, 1], jnp.full_like(p[:, 0], HALF)], -1)
  goal = jnp.where((t == 4)[:, None], table, on_block)
  goal_yaw = jnp.where(t == 4, _block_yaw(s['quat'][rows, h]),
                       _block_yaw(s['quat'][rows, jnp.minimum(t, 3)]))
  turn, _ = _turn(_block_yaw(s['quat'][rows, h]), goal_yaw)
  xy_ok = jnp.linalg.norm(goal[:, :2] - p[:, :2], axis=-1) < 0.005
  safe = jnp.maximum(p[:, 2], goal[:, 2] + 0.04)
  want_block = jnp.where(
      xy_ok[:, None], goal,
      jnp.stack([goal[:, 0], goal[:, 1], safe], -1))
  down = xy_ok & (jnp.abs(p[:, 2] - goal[:, 2]) < 0.004)
  want = s['ee'] + (want_block - p)
  want = jnp.where(holding[:, None], want,
                   s['ee'] + jnp.array([0.0, 0.0, 0.03]))
  grip = jnp.where(holding & ~down, 1.0, -1.0)
  turn = jnp.where(holding, turn, 0.0)
  return _act(s['ee'], want, turn, grip)
