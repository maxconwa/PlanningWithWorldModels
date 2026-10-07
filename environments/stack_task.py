"""Block-stacking task logic, as single-world JAX functions.

The tower prefix k counts goal blocks correctly in place from the bottom:
the first rests on the table, each next one on the one below it, and none
is held. The reward is the change in k each step, plus 1 on success, which
is k = N with the gripper open for more than HOLD_STEPS control steps.
Start layouts put blocks in distinct cells of a 4 x 4 grid with jitter, so
blocks never start closer than 7 cm.
"""

import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import COLORS, HALF, PARK

NONE = len(COLORS)  # Goal padding.
LIMITS = (0, 0, 400, 600, 800)  # Control steps per episode, by N.
HOLD_STEPS = 20  # 1 s at 20 Hz.
XY_TOL = 0.015
Z_TOL = 0.006
SLOTS = np.array([(0.35 + 0.1 * i, -0.15 + 0.1 * j)
                  for i in range(4) for j in range(4)])
JITTER = 0.015
GRIP_XY = 0.02  # How close a held block's centre is to the gripper site.
GRIP_Z = 0.03
GRIP_ON_BLOCK = (0.03, 0.05)  # Finger gap when closed on a block.


def sample_start(key, fixed_n, prestack):
  """N, the goal (padded with NONE), block positions and yaws. With
  probability `prestack`, 2 or 3 of the goal blocks start as one stack."""
  keys = jax.random.split(key, 8)
  n = jnp.where(fixed_n > 0, fixed_n, jax.random.randint(keys[0], (), 2, 5))
  levels = jnp.arange(len(COLORS))
  colors = jax.random.permutation(keys[1], len(COLORS))
  goal = jnp.where(levels < n, colors, NONE).astype(jnp.int32)
  stacked = jax.random.uniform(keys[2]) < prestack
  height = jnp.where(
      stacked, jax.random.randint(keys[3], (), 2, jnp.minimum(3, n) + 1), 1)
  # Each goal level gets a random rank; ranks below `height` share slot 0
  # as one stack, in rank order, and the rest get slots of their own.
  order = jnp.argsort(jnp.where(
      levels < n, jax.random.uniform(keys[4], (len(COLORS),)), 2.0))
  rank = jnp.argsort(order)
  in_stack = rank < height
  slots = jax.random.permutation(keys[5], len(SLOTS))
  slot = jnp.where(in_stack, slots[0], slots[rank - height + 1])
  xy = jnp.asarray(SLOTS)[slot] + jax.random.uniform(
      keys[6], (len(COLORS), 2), minval=-JITTER, maxval=JITTER)
  xy = jnp.where(in_stack[:, None], xy[jnp.argmin(rank)], xy)
  z = HALF + 2 * HALF * jnp.where(in_stack, rank, 0)
  yaw = jax.random.uniform(keys[7], (len(COLORS),),
                           minval=-np.pi / 4, maxval=np.pi / 4)
  yaw = jnp.where(in_stack, yaw[jnp.argmin(rank)], yaw)
  index = jnp.where(levels < n, goal, NONE)
  positions = jnp.asarray(PARK).at[index].set(
      jnp.concatenate([xy, z[:, None]], -1), mode='drop')
  yaws = jnp.zeros(len(COLORS)).at[index].set(yaw, mode='drop')
  return n.astype(jnp.int32), goal, positions, yaws


def held_mask(block_pos, site_pos, opening):
  """Which blocks sit between the fingers while they are closed on a
  block."""
  rel = block_pos - site_pos
  near = ((jnp.linalg.norm(rel[:, :2], axis=-1) < GRIP_XY) &
          (jnp.abs(rel[:, 2]) < GRIP_Z))
  closed = (opening > GRIP_ON_BLOCK[0]) & (opening < GRIP_ON_BLOCK[1])
  return near & closed


def tower_prefix(goal, n, block_pos, held):
  """How many goal blocks, from the bottom, are in place."""
  levels = jnp.arange(len(COLORS))
  idx = jnp.minimum(goal, len(COLORS) - 1)
  pos = block_pos[idx]
  below = jnp.concatenate([jnp.zeros((1, 3)), pos[:-1]])
  on_table = jnp.abs(pos[:, 2] - HALF) < Z_TOL
  on_below = ((jnp.linalg.norm(pos[:, :2] - below[:, :2], axis=-1) < XY_TOL) &
              (jnp.abs(pos[:, 2] - below[:, 2] - 2 * HALF) < Z_TOL))
  ok = jnp.where(levels == 0, on_table, on_below) & ~held[idx] & (levels < n)
  return jnp.cumprod(ok.astype(jnp.int32)).sum()


def progress(k_old, count, k, n, gripper_open):
  """The step's reward, whether the episode succeeded, and the hold count."""
  count = jnp.where((k == n) & gripper_open, count + 1, 0)
  success = count > HOLD_STEPS
  reward = (jnp.asarray(k) - k_old).astype(jnp.float32)
  return reward + success.astype(jnp.float32), success, count
