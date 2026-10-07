import time

import jax
import mujoco
import numpy as np
import pytest

from environments import stack_scene as scene
from environments.stack_scene import HALF
from environments.stack_sim import Sim


@pytest.fixture(scope='module')
def sim():
  return Sim(4)


def layout(c, pos, yaw=0.0, batch=4):
  """Every world with block c at `pos`, the rest parked."""
  positions = np.repeat(scene.PARK[None], batch, 0).astype(np.float32)
  positions[:, c] = pos
  yaws = np.zeros((batch, 4), np.float32)
  yaws[:, c] = yaw
  return jax.device_put(positions), jax.device_put(yaws)


def home_ctrl(batch=4):
  return jax.device_put(
      np.tile(scene.home_ctrl(), (batch, 1)).astype(np.float32))


def test_block_rests_on_floor(sim):
  d = sim.set_blocks(sim.make_data(), *layout(0, (0.5, 0.0, HALF + 0.01), 0.3))
  d = jax.jit(sim.physics, static_argnums=2)(d, home_ctrl(), 200)
  pos = np.asarray(jax.device_get(sim.block_pos(d)))
  assert np.allclose(pos[:, 0, 2], HALF, atol=2e-3)
  assert np.abs(np.asarray(jax.device_get(sim.block_vel(d)))).max() < 1e-2


def test_matches_cpu_mujoco(sim):
  """One second of a block dropping, against CPU MuJoCo."""
  d = sim.set_blocks(sim.make_data(), *layout(1, (0.45, 0.1, 0.08), 0.5))
  d = jax.jit(sim.physics, static_argnums=2)(d, home_ctrl(), 200)
  gpu = np.asarray(jax.device_get(sim.block_pos(d)))[0, 1]
  m = sim.mjm
  cpu = mujoco.MjData(m)
  cpu.qpos[:] = scene.home_qpos(m)
  a = scene.block_qpos_adr(m)[1]
  cpu.qpos[a:a + 7] = [0.45, 0.1, 0.08, np.cos(0.25), 0, 0, np.sin(0.25)]
  cpu.ctrl[:] = scene.home_ctrl()
  mujoco.mj_step(m, cpu, nstep=200)
  np.testing.assert_allclose(gpu, cpu.xpos[scene.block_body_ids(m)[1]],
                             atol=5e-3)


def test_both_cameras_render(sim):
  d = sim.set_blocks(sim.make_data(), *layout(0, (0.5, 0.0, HALF)))
  workspace, wrist, _ = jax.jit(sim.render)(d)
  for image in (workspace, wrist):
    image = np.asarray(jax.device_get(image))
    assert image.shape == (4, 64, 64, 3) and image.dtype == np.uint8
    assert image.std() > 5


def test_workspace_camera_sees_a_block(sim):
  d = sim.set_blocks(sim.make_data(), *layout(0, (0.5, 0.0, HALF)))
  workspace, _, _ = jax.jit(sim.render)(d)
  image = np.asarray(jax.device_get(workspace))[0].astype(int)
  red = (image[..., 0] > 150) & (image[..., 1] < 80) & (image[..., 2] < 80)
  assert red.sum() >= 4


def test_throughput():
  """Prints control steps per second with rendering for 256 worlds."""
  sim = Sim(256)
  ctrl = home_ctrl(256)

  @jax.jit
  def step(d):
    d = sim.physics(d, ctrl, scene.SUBSTEPS)
    workspace, wrist, d = sim.render(d)
    return d, workspace.sum() + wrist.sum()

  d, total = step(sim.make_data())
  jax.block_until_ready(total)
  start = time.time()
  for _ in range(20):
    d, total = step(d)
  jax.block_until_ready(total)
  rate = 20 * 256 / (time.time() - start)
  print(f'\n{rate:.0f} control steps/s with two cameras, 256 worlds')
  assert rate > 0


def test_large_batch_has_room_for_every_contact():
  """Four blocks on the floor in each of 64 worlds stay at rest."""
  sim = Sim(64)
  positions = np.zeros((64, 4, 3), np.float32)
  for c in range(4):
    positions[:, c] = (0.4 + 0.08 * c, 0.0, HALF)
  d = sim.set_blocks(sim.make_data(), jax.device_put(positions),
                     jax.device_put(np.zeros((64, 4), np.float32)))
  d = jax.jit(sim.physics, static_argnums=2)(d, home_ctrl(64), 200)
  pos = np.asarray(jax.device_get(sim.block_pos(d)))
  np.testing.assert_allclose(pos, positions, atol=2e-3)
