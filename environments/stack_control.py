"""End-effector control for the Panda: 5-D actions to joint targets, in JAX.

An action moves a target for the gripper site by up to MAX_STEP per control
step, turns its yaw by up to MAX_YAW, and opens or closes the gripper. The
gripper always points down. Damped least-squares IK turns the target into
joint position targets for the arm's actuators; its Jacobian comes from
differentiating MJX's (JAX backend) forward kinematics.
"""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from environments import stack_scene as scene

LO = np.array([0.30, -0.22, 0.015])  # Gripper site bounds, metres.
HI = np.array([0.70, 0.22, 0.35])
MAX_STEP = 0.02
MAX_YAW = 0.2
YAW_LIMIT = np.pi / 2
HOLD = (0, 0, 0, 0, -1)  # Stay put with the gripper open.


def rot_z(angle):
  c, s = jnp.cos(angle), jnp.sin(angle)
  zero, one = jnp.zeros_like(c), jnp.ones_like(c)
  return jnp.stack([jnp.stack([c, -s, zero]), jnp.stack([s, c, zero]),
                    jnp.stack([zero, zero, one])])


class Controller:

  def __init__(self, sim, iters=5, damping=0.05):
    m = sim.mjm
    self.site = sim.site
    self.iters, self.damping = iters, damping
    self.kin_model = mjx.put_model(m)
    self.kin_data = mjx.make_data(m)
    home = mujoco.MjData(m)
    home.qpos[:] = scene.home_qpos(m)
    mujoco.mj_kinematics(m, home)
    # Yaw is measured from the gripper's orientation at the home pose.
    self.home_mat = jax.device_put(
        home.site_xmat[self.site].reshape(3, 3).astype(np.float32))
    self.ranges = jax.device_put(m.jnt_range[:7].astype(np.float32))
    self.lo = jax.device_put(LO.astype(np.float32))
    self.hi = jax.device_put(HI.astype(np.float32))

  def pose(self, qpos):
    """Gripper site position and orientation for one world's qpos."""
    d = mjx.kinematics(self.kin_model, self.kin_data.replace(qpos=qpos))
    return d.site_xpos[self.site], d.site_xmat[self.site].reshape(3, 3)

  def ee(self, d):
    pos = d.site_xpos[:, self.site]
    mat = d.site_xmat[:, self.site].reshape(-1, 3, 3)
    rel = mat @ self.home_mat.T
    return pos, jnp.arctan2(rel[:, 1, 0], rel[:, 0, 0])

  def start(self, d):
    pos, yaw = self.ee(d)
    return (jnp.clip(pos, self.lo, self.hi),
            jnp.clip(yaw, -YAW_LIMIT, YAW_LIMIT))

  def act(self, d, target, yaw, action):
    """Joint and finger targets for one control step, and the new
    end-effector targets."""
    a = jnp.clip(action, -1, 1)
    target = jnp.clip(target + MAX_STEP * a[:, :3], self.lo, self.hi)
    yaw = jnp.clip(yaw + MAX_YAW * a[:, 3], -YAW_LIMIT, YAW_LIMIT)
    mats = jax.vmap(rot_z)(yaw) @ self.home_mat
    arm = jax.vmap(self.solve)(d.qpos, target, mats)
    fingers = jnp.where(a[:, 4] > 0, scene.CLOSED, scene.OPEN)
    return jnp.concatenate([arm, fingers[:, None]], -1), target, yaw

  def solve(self, qpos, pos, mat):
    """Arm joint angles that put the gripper site at `pos` with
    orientation `mat`, for one world."""

    def error(q):
      cur_pos, cur = self.pose(qpos.at[:7].set(q))
      # Small-angle rotation from the current to the target orientation.
      rot = 0.5 * jnp.cross(cur.T, mat.T).sum(0)
      return jnp.concatenate([pos - cur_pos, rot])

    def body(_, q):
      err, jac = error(q), jax.jacfwd(error)(q)
      lhs = jac @ jac.T + self.damping ** 2 * jnp.eye(6)
      q = q - jac.T @ jnp.linalg.solve(lhs, err)
      return jnp.clip(q, self.ranges[:, 0], self.ranges[:, 1])

    return jax.lax.fori_loop(0, self.iters, body, qpos[:7])
