import jax
import numpy as np
import pytest

from environments import stack_scene as scene
from environments.stack_control import HI, LO, MAX_STEP, Controller
from environments.stack_scene import HALF
from environments.stack_sim import Sim


class Arm:
  """One world driven through the controller."""

  def __init__(self):
    self.sim = Sim(1)
    self.ctrl = Controller(self.sim)
    self.d = self.sim.make_data()
    self.target, self.yaw = jax.jit(self.ctrl.start)(self.d)

    @jax.jit
    def step(d, target, yaw, action):
      ctrl, target, yaw = self.ctrl.act(d, target, yaw, action)
      return self.sim.physics(d, ctrl, scene.SUBSTEPS), target, yaw
    self._step = step

  def apply(self, action):
    action = jax.device_put(np.asarray(action, np.float32)[None])
    self.d, self.target, self.yaw = self._step(
        self.d, self.target, self.yaw, action)

  def pos(self):
    return np.asarray(jax.device_get(self.ctrl.ee(self.d)[0]))[0]

  def yaw_now(self):
    return float(jax.device_get(self.ctrl.ee(self.d)[1])[0])

  def settle(self, steps=20):
    for _ in range(steps):
      self.apply([0, 0, 0, 0, -1])

  def move_to(self, pos, grip, steps=60):
    for _ in range(steps):
      target = np.asarray(jax.device_get(self.target))[0]
      delta = np.clip((np.asarray(pos) - target) / MAX_STEP, -1, 1)
      self.apply(np.concatenate([delta, [0.0, grip]]))


def test_start_target_is_inside_the_bounds():
  target = np.asarray(jax.device_get(Arm().target))[0]
  assert (target >= LO).all() and (target <= HI).all()


def test_tracks_its_target():
  """The gripper settles within 5 mm of its target, against gravity."""
  arm = Arm()
  arm.move_to((0.5, 0.1, 0.15), -1)
  arm.settle(30)
  target = np.asarray(jax.device_get(arm.target))[0]
  np.testing.assert_allclose(arm.pos(), target, atol=5e-3)


def test_moves_along_x():
  # Home puts the gripper near the far x bound and 3 cm above the top one,
  # so let it come down first, then move towards the base.
  arm = Arm()
  arm.settle(60)
  start = arm.pos()
  for _ in range(5):
    arm.apply([-1.0, 0, 0, 0, -1])
  arm.settle(10)
  moved = arm.pos() - start
  assert moved[0] == pytest.approx(-0.10, abs=0.01)
  assert abs(moved[1]) < 0.01 and abs(moved[2]) < 0.01


def test_turns_yaw():
  arm = Arm()
  arm.settle()
  start = arm.yaw_now()
  for _ in range(4):
    arm.apply([0, 0, 0, 1.0, -1])
  arm.settle(10)
  assert arm.yaw_now() - start == pytest.approx(0.8, abs=0.05)


def test_random_actions_stay_bounded():
  arm = Arm()
  rng = np.random.default_rng(0)
  for _ in range(300):
    arm.apply(rng.uniform(-1, 1, 5))
    pos = arm.pos()
    assert np.isfinite(pos).all()
    assert (pos > LO - 0.03).all() and (pos < HI + 0.03).all()


def test_grasps_and_lifts_a_block():
  arm = Arm()
  positions = scene.PARK[None].astype(np.float32).copy()
  positions[0, 0] = (0.5, 0.0, HALF)
  arm.d = arm.sim.set_blocks(arm.d, jax.device_put(positions),
                             jax.device_put(np.zeros((1, 4), np.float32)))
  arm.move_to((0.5, 0.0, 0.12), -1)
  arm.move_to((0.5, 0.0, HALF), -1)
  arm.move_to((0.5, 0.0, HALF), 1, steps=15)
  arm.move_to((0.5, 0.0, 0.15), 1)
  z = float(np.asarray(jax.device_get(arm.sim.block_pos(arm.d)))[0, 0, 2])
  assert z > 0.10


def test_more_iterations_reach_a_far_target():
  """Placing the arm for a start needs IK from home to far away."""
  import jax.numpy as jnp
  from environments.stack_control import rot_z
  arm = Arm()
  goal = jnp.array([0.45, -0.1, 0.2])
  mat = rot_z(jnp.array(0.5)) @ arm.ctrl.home_mat
  qpos = arm.d.qpos[0]
  q = jax.jit(lambda: arm.ctrl.solve(qpos, goal, mat, iters=30))()
  pos, _ = arm.ctrl.pose(qpos.at[:7].set(q))
  np.testing.assert_allclose(np.asarray(pos), np.asarray(goal), atol=2e-3)
