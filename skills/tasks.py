"""Pick and place as single-world JAX functions: which blocks can be
picked, how far a block is lifted, whether a placement is done, start
states, and the shaped potentials the teachers learn from.

A teacher's reward is the change in its potential each step, plus
SUCCESS_BONUS on the step the skill succeeds, which ends it. The potentials
are for training only; the stacking agents never see them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import COLORS, HALF, OPEN_WIDTH
from environments.stack_task import (
    NONE, SLOTS, XY_TOL, Z_TOL, sample_start)

EPISODE = 150  # Control steps before a skill times out.
LIFT = 0.05  # How far a picked block must clear what is under it.
# The pick potential keeps rising to here, past LIFT, so a policy lifts
# clear of the success line rather than hovering on it.
LIFT_REWARD = 0.07
RAISE = 0.05  # How far the gripper rises above a placed block.
TABLE = NONE  # Place target index for the table.
SUCCESS_BONUS = 5.0
NEAR_XY = 0.03  # Blocks closer than this horizontally are in one stack.
FREE = 0.06  # Table cells closer than this to another block are taken.
ARM_LO = (0.35, -0.18)  # Gripper xy range of starts away from home.
ARM_HI = (0.65, 0.18)
SITE_HI = (0.65, 0.18, 0.22)  # Highest a place start holds the block.
HOME_SHARE = 0.5  # Share of pick starts with the arm at home.

_IDX = jnp.arange(len(COLORS))
_SLOTS = jnp.asarray(SLOTS, jnp.float32)


def clear_mask(pos, present):
  """Present blocks with no other present block on top of them."""
  rel = pos[None] - pos[:, None]  # [c, o]: block o relative to block c.
  others = present[None] & ~jnp.eye(len(COLORS), dtype=bool)
  on_top = ((jnp.linalg.norm(rel[..., :2], axis=-1) < NEAR_XY) &
            (rel[..., 2] > HALF) & others)
  return present & ~on_top.any(1)


def clearance(pos, present, c):
  """The gap between block c's bottom and the top of whatever is under it
  (another block or the table)."""
  p = pos[c]
  under = ((jnp.linalg.norm(pos[:, :2] - p[:2], axis=-1) < 2 * HALF) &
           present & (_IDX != c) & (pos[:, 2] < p[2]))
  support = jnp.max(jnp.where(under, pos[:, 2] + HALF, 0.0))
  return p[2] - HALF - support


def resting_on(pos, h, t):
  """Whether block h rests on block t, or on the table when t is TABLE."""
  p, q = pos[h], pos[jnp.minimum(t, len(COLORS) - 1)]
  on_table = jnp.abs(p[2] - HALF) < Z_TOL
  on_block = ((jnp.linalg.norm(p[:2] - q[:2]) < XY_TOL) &
              (jnp.abs(p[2] - q[2] - 2 * HALF) < Z_TOL))
  return jnp.where(t == TABLE, on_table, on_block)


def top_under(pos, present, xy):
  """The highest top of the present blocks within FREE of xy, or 0."""
  near = (jnp.linalg.norm(pos[:, :2] - xy, axis=-1) < FREE) & present
  return jnp.max(jnp.where(near, pos[:, 2] + HALF, 0.0))


def table_spot(pos, present, h):
  """Where block h goes on the table: straight down when that spot is at
  least FREE from every other present block, else the nearest such cell of
  the start grid."""
  others = present & (_IDX != h)
  own = jnp.where(others, jnp.linalg.norm(
      pos[:, :2] - pos[h, :2], axis=-1) >= FREE, True).all()
  dist = jnp.linalg.norm(_SLOTS[:, None] - pos[None, :, :2], axis=-1)
  free = jnp.where(others[None], dist >= FREE, True).all(1)
  near = jnp.linalg.norm(_SLOTS - pos[h, :2], axis=-1)
  cell = _SLOTS[jnp.argmin(jnp.where(free, near, jnp.inf))]
  xy = jnp.where(own, pos[h, :2], cell)
  return jnp.concatenate([xy, jnp.full((1,), HALF, pos.dtype)])


def place_goal(pos, present, h, t):
  """Where block h's centre goes: on top of block t, or a free spot on the
  table."""
  q = pos[jnp.minimum(t, len(COLORS) - 1)]
  return jnp.where(t == TABLE, table_spot(pos, present, h),
                   q + jnp.array([0.0, 0.0, 2 * HALF]))


def pick_progress(pos, present, held, site, c):
  """Potential and success for picking block c: reach it, grasp it, lift
  it LIFT clear."""
  grasped = held[c].astype(jnp.float32)
  lift = jnp.clip(clearance(pos, present, c) / LIFT_REWARD, 0.0, 1.0)
  phi = -jnp.linalg.norm(site - pos[c]) + grasped + 2.0 * grasped * lift
  return phi, held[c] & (clearance(pos, present, c) >= LIFT)


def place_progress(pos, present, held, site, opening, h, t):
  """Potential and success for placing block h on t: bring it to the
  goal, let go, open, and raise the gripper RAISE above it."""
  goal = place_goal(pos, present, h, t)
  resting = ~held[h] & resting_on(pos, h, t)
  opened = opening > OPEN_WIDTH
  raised = site[2] - (pos[h, 2] + HALF)
  done = resting & opened
  phi = (-2.0 * jnp.linalg.norm(pos[h] - goal) + resting.astype(jnp.float32)
         + done.astype(jnp.float32) * jnp.clip(raised / RAISE, 0.0, 1.0))
  return phi, done & (raised >= RAISE)


def _choose(key, mask):
  """A random index where `mask` is true."""
  return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))


def _present(goal):
  return (_IDX[:, None] == goal[None, :]).any(1)


def pick_start(key, prestack, stacked=0.0):
  """A stacking layout and a clear block to pick. With probability
  `stacked` the block is one resting on another, when there is one."""
  layout_key, choice_key, coin = jax.random.split(key, 3)
  _, goal, pos, yaws = sample_start(layout_key, 0, prestack)
  present = _present(goal)
  clear = clear_mask(pos, present)
  on_block = clear & (pos[:, 2] > HALF + Z_TOL)
  favour = (jax.random.uniform(coin) < stacked) & on_block.any()
  target = _choose(choice_key, jnp.where(favour, on_block, clear))
  return present, pos, yaws, target


def arm_start(key, pos, present):
  """A gripper site and yaw over the workspace, RAISE or more above the
  blocks under it, as a place leaves the arm."""
  k1, k2, k3 = jax.random.split(key, 3)
  xy = jax.random.uniform(k1, (2,), minval=jnp.array(ARM_LO),
                          maxval=jnp.array(ARM_HI))
  z = jnp.maximum(top_under(pos, present, xy), 2 * HALF) + RAISE
  z = z + jax.random.uniform(k2, (), maxval=0.15)
  yaw = jax.random.uniform(k3, (), minval=-np.pi / 4, maxval=np.pi / 4)
  return jnp.concatenate([xy, z[None]]), yaw


def place_start(key, prestack):
  """A stacking layout with one clear block moved into the gripper, held
  from just clear of the blocks under it (as a pick leaves it) up to
  SITE_HI, and a place target for it: another clear block or the
  table."""
  keys = jax.random.split(key, 5)
  _, goal, pos, yaws = sample_start(keys[0], 0, prestack)
  present = _present(goal)
  held = _choose(keys[1], clear_mask(pos, present))
  rest = present & (_IDX != held)
  xy = jax.random.uniform(keys[2], (2,), minval=jnp.array(ARM_LO),
                          maxval=jnp.array(ARM_HI))
  low = top_under(pos, rest, xy) + HALF + 0.01
  z = jax.random.uniform(jax.random.fold_in(keys[2], 1), (), minval=low,
                         maxval=jnp.maximum(low, SITE_HI[2]))
  site = jnp.concatenate([xy, z[None]])
  yaw = jax.random.uniform(keys[3], (), minval=-np.pi / 4, maxval=np.pi / 4)
  pos = pos.at[held].set(site)
  yaws = yaws.at[held].set(yaw)
  options = jnp.concatenate([clear_mask(pos, rest), jnp.array([True])])
  target = _choose(keys[4], options)
  return present, pos, yaws, held, target, site, yaw
