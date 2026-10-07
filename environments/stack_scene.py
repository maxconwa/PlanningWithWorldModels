"""The block-stacking model: a Franka Panda, four blocks and two cameras.

The Panda is Menagerie's MJX model (assets/panda/). Blocks not in the
current episode are parked behind the workspace camera, out of every view.
"""

import pathlib

import mujoco
import numpy as np

ASSETS = pathlib.Path(__file__).resolve().parent / 'assets'
COLORS = ('red', 'green', 'blue', 'yellow')
HALF = 0.02  # Half the block edge.
SUBSTEPS = 10  # Physics steps per control step.
CONTROL_DT = 0.05
HOME_ARM = np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853])
OPEN, CLOSED = 0.04, 0.0  # Finger targets.
OPEN_WIDTH = 0.07  # Finger gap above which the gripper counts as open.
ARM_BODIES = tuple(f'link{i}' for i in range(1, 8)) + (
    'hand', 'left_finger', 'right_finger')
PARK = np.array([[1.5, -0.3 + 0.2 * c, HALF] for c in range(len(COLORS))])


def build_model():
  """The scene with the wrist camera added to the Panda's hand."""
  spec = mujoco.MjSpec.from_file(str(ASSETS / 'stack.xml'))
  # The included Panda's meshes live in panda/assets/, which its own
  # meshdir names relative to itself; MjSpec loses that through the include.
  spec.meshdir = str(ASSETS / 'panda' / 'assets')
  for mesh in spec.meshes:
    mesh.file = pathlib.Path(mesh.file).name
  # Gravity compensation on the arm, as the real Panda's controller does;
  # without it the position servos sag a few centimetres below target.
  for name in ARM_BODIES:
    spec.body(name).gravcomp = 1
  # In the hand frame z points from the wrist to the fingertips; turning the
  # camera half a turn about x makes it look along z, past the fingers.
  spec.body('hand').add_camera(
      name='wrist', pos=[0.06, 0, 0.0], quat=[0, 1, 0, 0], fovy=75)
  return spec.compile()


def block_qpos_adr(model):
  return [int(model.jnt_qposadr[model.joint(f'block_{c}').id])
          for c in COLORS]


def block_dof_adr(model):
  return [int(model.jnt_dofadr[model.joint(f'block_{c}').id])
          for c in COLORS]


def block_body_ids(model):
  return [model.body(f'block_{c}').id for c in COLORS]


def home_qpos(model):
  """Arm at home, gripper open, every block parked."""
  qpos = model.qpos0.copy()
  qpos[:7] = HOME_ARM
  qpos[7:9] = OPEN
  for c, a in enumerate(block_qpos_adr(model)):
    qpos[a:a + 3] = PARK[c]
    qpos[a + 3:a + 7] = [1, 0, 0, 0]
  return qpos


def home_ctrl():
  return np.concatenate([HOME_ARM, [OPEN]])
