"""Pick and place as single-world JAX functions: which blocks can be
picked, how far a block is lifted, whether a placement is done, start
states, and the shaped potentials the teachers learn from.

A teacher's reward is the change in its potential each step, plus
SUCCESS_BONUS when it ends the skill having succeeded, or -FAIL_PENALTY when
it ends it without. The potentials are for training only; the stacking agents
never see them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import COLORS, HALF, OPEN_WIDTH
from environments.stack_task import NONE, XY_TOL, Z_TOL, sample_start

EPISODE = 150  # Control steps before a skill times out.
LIFT = 0.05  # How far a picked block must clear what is under it.
RAISE = 0.05  # How far the gripper rises above a placed block.
TABLE = NONE  # Place target index for the table.
SUCCESS_BONUS = 5.0
FAIL_PENALTY = 1.0
NEAR_XY = 0.03  # Blocks closer than this horizontally are in one stack.
SITE_LO = (0.40, -0.15, 0.17)  # Where place starts hold the block.
SITE_HI = (0.60, 0.15, 0.22)

_IDX = jnp.arange(len(COLORS))


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


def place_goal(pos, h, t):
  """Where block h's centre goes: on top of block t, or straight down onto
  the table."""
  p, q = pos[h], pos[jnp.minimum(t, len(COLORS) - 1)]
  table = jnp.stack([p[0], p[1], jnp.asarray(HALF, p.dtype)])
  return jnp.where(t == TABLE, table, q + jnp.array([0.0, 0.0, 2 * HALF]))


def pick_progress(pos, present, held, site, c):
  """Potential and success for picking block c: reach it, grasp it, lift
  it LIFT clear."""
  grasped = held[c].astype(jnp.float32)
  lift = jnp.clip(clearance(pos, present, c) / LIFT, 0.0, 1.0)
  phi = -jnp.linalg.norm(site - pos[c]) + grasped + 2.0 * grasped * lift
  return phi, held[c] & (clearance(pos, present, c) >= LIFT)


def place_progress(pos, held, site, opening, h, t):
  """Potential and success for placing block h on t: bring it to the
  goal, let go, open, and raise the gripper RAISE above it."""
  goal = place_goal(pos, h, t)
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


def pick_start(key, prestack):
  """A stacking layout and a clear block to pick."""
  layout_key, choice_key = jax.random.split(key)
  _, goal, pos, yaws = sample_start(layout_key, 0, prestack)
  present = _present(goal)
  target = _choose(choice_key, clear_mask(pos, present))
  return present, pos, yaws, target


def place_start(key, prestack):
  """A stacking layout with one clear block moved into the gripper, held
  above everything else, and a place target for it: another clear block or
  the table."""
  keys = jax.random.split(key, 5)
  _, goal, pos, yaws = sample_start(keys[0], 0, prestack)
  present = _present(goal)
  held = _choose(keys[1], clear_mask(pos, present))
  site = jax.random.uniform(keys[2], (3,), minval=jnp.array(SITE_LO),
                            maxval=jnp.array(SITE_HI))
  yaw = jax.random.uniform(keys[3], (), minval=-np.pi / 4, maxval=np.pi / 4)
  pos = pos.at[held].set(site)
  yaws = yaws.at[held].set(yaw)
  rest = present & (_IDX != held)
  options = jnp.concatenate([clear_mask(pos, rest), jnp.array([True])])
  target = _choose(keys[4], options)
  return present, pos, yaws, held, target, site, yaw
