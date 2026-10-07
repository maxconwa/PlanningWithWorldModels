import mujoco
import numpy as np
import pytest

from environments import stack_scene as scene
from environments.stack_scene import HALF


@pytest.fixture(scope='module')
def model():
  return scene.build_model()


def settle(model, qpos, steps=200):
  data = mujoco.MjData(model)
  data.qpos[:] = qpos
  data.ctrl[:] = scene.home_ctrl()
  mujoco.mj_step(model, data, nstep=steps)
  return data


def place(model, qpos, c, pos, yaw=0.0):
  a = scene.block_qpos_adr(model)[c]
  qpos[a:a + 3] = pos
  qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]


def test_scene_has_blocks_cameras_and_timing(model):
  for c in scene.COLORS:
    assert model.body(f'block_{c}').id >= 0
  assert model.camera('workspace').id >= 0
  assert model.camera('wrist').id >= 0
  assert model.opt.timestep * scene.SUBSTEPS == pytest.approx(
      scene.CONTROL_DT)


def test_home_parks_every_block(model):
  data = settle(model, scene.home_qpos(model), 1)
  for b in scene.block_body_ids(model):
    assert data.xpos[b][0] > 1.2


def test_block_rests_on_floor(model):
  qpos = scene.home_qpos(model)
  place(model, qpos, 0, (0.5, 0.0, HALF + 0.01), 0.3)
  data = settle(model, qpos)
  b = scene.block_body_ids(model)[0]
  assert data.xpos[b][2] == pytest.approx(HALF, abs=2e-3)


def test_blocks_stack_on_each_other(model):
  qpos = scene.home_qpos(model)
  place(model, qpos, 0, (0.5, 0.0, HALF))
  place(model, qpos, 1, (0.5, 0.0, 3 * HALF + 0.005))
  data = settle(model, qpos)
  b = scene.block_body_ids(model)[1]
  assert data.xpos[b][2] == pytest.approx(3 * HALF, abs=3e-3)
