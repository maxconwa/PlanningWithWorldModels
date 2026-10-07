# Block Stacking Environment Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A MuJoCo Panda block-stacking environment with raw end-effector
actions and two 64 x 64 cameras, registered as `custom_stack`, that DreamerV3
trains on through `train.py`.

**Architecture:** Four small modules in `environments/`. `stack_scene.py`
builds the MuJoCo model (Menagerie's MJX Panda, a floor, four blocks, a
workspace camera and a wrist camera) and wraps model and data. `stack_control.py`
turns 5-D end-effector actions into joint targets by damped least-squares IK.
`stack_task.py` holds the pure task logic (start states, tower prefix,
reward and success) with no MuJoCo in it. `stack.py` is the `embodied.Env`
that joins them. Tests live in `tests/` and run with pytest.

**Tech Stack:** Python 3.11, `mujoco` 3.15.0 (CPU, EGL rendering),
`mujoco-mjx` 3.15.0 (only checked to load here), NumPy, DreamerV3's
`embodied`/`elements`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-block-stacking-options-design.md`

## Global Constraints

- Python with 2-space indentation, single quotes, lines of about 80 columns,
  no type hints; short docstrings saying what a function is for; comments
  describe the code as it is (AGENTS.md "Code style").
- Do not edit `third_party/` (AGENTS.md "Layout").
- Run everything with the `dreamerv3` conda env:
  `PY=~/miniconda3/envs/dreamerv3/bin/python`. JAX stays at 0.10.2; installs
  use `-c constraints.txt` so pip cannot move it.
- Only one GPU process at a time (AGENTS.md "Python environments"). Tests run
  on CPU; check `nvidia-smi` before any DreamerV3 run.
- Four colours, in this order everywhere: red, green, blue, yellow (indices
  0-3); `NONE = 4` pads the goal.
- Blocks are 4 cm cubes (`HALF = 0.02`).
- Physics timestep 0.005 s, 10 physics steps per control step, control at
  20 Hz.
- Raw action: 5 floats in [-1, 1]: dx, dy, dz (up to 0.02 m per step), dyaw
  (up to 0.2 rad per step), gripper (> 0 closes).
- Time limits in control steps: N = 2: 400, N = 3: 600, N = 4: 800.
- Success: tower prefix k = N with the gripper open, held for 20 control
  steps (1 s).
- Reward: change in k each step, plus 1 on success.
- 30% of starts pre-stack 2 or 3 of the blocks.
- `log/` observation keys must be scalars (DreamerV3's
  `embodied/run/train.py:42` asserts it). Arrays go through `privileged()`.

## Review Focus

1. **Blocks spawning inside each other or a stack collapsing at reset.** A
   reasonable person expects every episode to start with blocks at rest.
   Test: Task 6, `test_reset_starts_at_rest` over 30 seeds.
2. **Random actions driving the arm into the floor, out of reach, or into
   NaNs.** Expected: the end effector stays inside the bounds and every
   observation stays finite. Test: Task 3, `test_random_actions_stay_bounded`.
3. **A built tower being knocked over.** Expected: k drops and the reward is
   negative on that step; the episode goes on. Test: Task 6,
   `test_toppling_gives_negative_reward`.
4. **The time limit arriving while a finished tower is being held for its
   1 s.** Expected: `is_last` without `is_terminal`, no success reward. Test:
   Task 5, `test_timeout_before_hold_is_not_success`, and Task 6,
   `test_time_limit_ends_episode`.
5. **Rendering in a headless subprocess** (how DreamerV3 runs environments).
   Expected: EGL is chosen without a display. Test: Task 4,
   `test_renders_in_fresh_headless_process`.

## File Structure

- Create `constraints.txt`: pins `jax==0.10.2` and `jaxlib==0.10.2` for pip.
- Create `environments/assets/panda/`: Menagerie's MJX Panda (`mjx_panda.xml`,
  `assets/`, `LICENSE`, `README.md` naming the source commit).
- Create `environments/assets/stack.xml`: floor, lights, blocks, workspace
  camera; includes `panda/mjx_panda.xml`.
- Create `environments/stack_scene.py`: `build_model()`, `Scene`.
- Create `environments/stack_control.py`: `Controller`.
- Create `environments/stack_task.py`: `Start`, `sample_start`,
  `tower_prefix`, `Progress`, constants.
- Create `environments/stack.py`: `Stack(embodied.Env)`.
- Modify `environments/__init__.py`: register `stack`.
- Modify `environments/play.py`: continuous keys, several camera panels.
- Modify `environments/README.md`: document `stack`.
- Create `tests/conftest.py`, `tests/test_stack_scene.py`,
  `tests/test_stack_control.py`, `tests/test_stack_render.py`,
  `tests/test_stack_task.py`, `tests/test_stack_env.py`.

---

### Task 1: Dependencies and the Panda model

**Files:**
- Create: `constraints.txt`
- Create: `environments/assets/panda/` (copied from Menagerie)
- Create: `tests/conftest.py`
- Test: `tests/test_stack_scene.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `environments/assets/panda/mjx_panda.xml` with body `hand`, site
  `gripper`, bodies `left_finger`/`right_finger`, actuators `actuator1`-`8`;
  pytest runnable as `$PY -m pytest tests`.

- [ ] **Step 1: Pin JAX and install the packages**

```bash
cd /home/max/PlanningWithWorldModels
cat > constraints.txt <<'EOF'
# Versions pip must not move when installing into the dreamerv3 env.
jax==0.10.2
jaxlib==0.10.2
EOF
PY=~/miniconda3/envs/dreamerv3/bin/python
$PY -m pip install -c constraints.txt mujoco==3.15.0 mujoco-mjx==3.15.0 pytest
$PY -c "import jax, mujoco; from mujoco import mjx; print(jax.__version__, mujoco.__version__)"
```

Expected output ends with `0.10.2 3.15.0`.

- [ ] **Step 2: Copy the Panda from Menagerie**

```bash
cd /home/max/PlanningWithWorldModels
tmp=$(mktemp -d)
git clone -q --depth 1 --filter=blob:none --sparse \
  https://github.com/google-deepmind/mujoco_menagerie.git $tmp/m
git -C $tmp/m sparse-checkout set franka_emika_panda
commit=$(git -C $tmp/m rev-parse HEAD)
mkdir -p environments/assets/panda
cp -r $tmp/m/franka_emika_panda/{mjx_panda.xml,assets,LICENSE} \
  environments/assets/panda/
cat > environments/assets/panda/README.md <<EOF
# Franka Emika Panda

\`mjx_panda.xml\` and \`assets/\` are copied unchanged from MuJoCo Menagerie
(\`franka_emika_panda/\`, commit $commit) under its Apache 2.0 licence
(\`LICENSE\`). The wrist camera is added in code by
\`environments/stack_scene.py\`, so this copy stays identical to upstream.
EOF
rm -rf $tmp
```

- [ ] **Step 3: Write the test harness and a failing load test**

`tests/conftest.py`:

```python
"""Puts the repository and the DreamerV3 checkout on sys.path, and makes
MuJoCo render with EGL, as the environments do in DreamerV3's subprocesses."""

import os
import pathlib
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
```

`tests/test_stack_scene.py`:

```python
import mujoco

from environments import stack_scene


def test_panda_model_compiles():
  path = stack_scene.ASSETS / 'panda' / 'mjx_panda.xml'
  model = mujoco.MjModel.from_xml_path(str(path))
  assert model.nu == 8
  assert model.site('gripper').id >= 0
```

- [ ] **Step 4: Run it to see it fail**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: FAIL with `ImportError: cannot import name 'stack_scene'`.

- [ ] **Step 5: Create the module with its asset path**

`environments/stack_scene.py`:

```python
"""The MuJoCo scene for block stacking: a Franka Panda, four blocks and two
cameras."""

import os

# EGL renders without a display, as DreamerV3's environment subprocesses
# need; set before mujoco is imported.
os.environ.setdefault('MUJOCO_GL', 'egl')

import pathlib  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parent / 'assets'
```

- [ ] **Step 6: Run the test to see it pass**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: `1 passed`.

- [ ] **Step 7: Commit**

```bash
git add constraints.txt environments/assets/panda environments/stack_scene.py tests
git commit -m "Add the Menagerie Panda and a test harness for block stacking"
```

---

### Task 2: The stacking scene

**Files:**
- Create: `environments/assets/stack.xml`
- Modify: `environments/stack_scene.py`
- Test: `tests/test_stack_scene.py`

**Interfaces:**
- Consumes: `environments/assets/panda/mjx_panda.xml` (Task 1).
- Produces, in `environments/stack_scene.py`:
  - constants `COLORS = ('red', 'green', 'blue', 'yellow')`, `HALF = 0.02`,
    `SUBSTEPS = 10`, `CONTROL_DT = 0.05`, `HOME_ARM` (7 floats),
    `OPEN = 0.04`, `CLOSED = 0.0`, `OPEN_WIDTH = 0.07`
  - `build_model() -> mujoco.MjModel`
  - `class Scene(size=(64, 64))` with attributes `model`, `data`, `site`
    (gripper site id) and methods `reset_arm()`, `set_block(c, pos, yaw)`,
    `park_block(c)`, `forward()`, `step_physics(n)`, `positions() -> (4, 3)`,
    `quats() -> (4, 4)`, `velocities() -> (4, 3)`, `held() -> set of int`,
    `gripper_opening() -> float`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stack_scene.py`:

```python
import numpy as np
import pytest

from environments.stack_scene import HALF, Scene


@pytest.fixture
def scene():
  return Scene()


def test_scene_has_blocks_and_cameras(scene):
  m = scene.model
  for c in stack_scene.COLORS:
    assert m.body(f'block_{c}').id >= 0
  assert m.camera('workspace').id >= 0
  assert m.camera('wrist').id >= 0
  assert m.opt.timestep * stack_scene.SUBSTEPS == pytest.approx(
      stack_scene.CONTROL_DT)


def test_block_rests_on_floor(scene):
  scene.set_block(0, (0.5, 0.0, HALF + 0.01), 0.3)
  scene.step_physics(200)
  pos = scene.positions()[0]
  assert pos[2] == pytest.approx(HALF, abs=2e-3)
  assert np.abs(scene.velocities()[0]).max() < 1e-2


def test_blocks_stack_on_each_other(scene):
  scene.set_block(0, (0.5, 0.0, HALF), 0.0)
  scene.set_block(1, (0.5, 0.0, 3 * HALF + 0.005), 0.0)
  scene.step_physics(200)
  assert scene.positions()[1][2] == pytest.approx(3 * HALF, abs=3e-3)


def test_parked_blocks_are_out_of_the_workspace(scene):
  scene.reset_arm()
  for pos in scene.positions():
    assert pos[0] > 1.2


def test_nothing_held_at_home(scene):
  scene.reset_arm()
  scene.step_physics(20)
  assert scene.held() == set()
  assert scene.gripper_opening() > stack_scene.OPEN_WIDTH


def test_mjx_loads_the_same_model():
  from mujoco import mjx
  mx = mjx.put_model(stack_scene.build_model())
  dx = mjx.make_data(mx)
  dx = mjx.step(mx, dx)
  assert dx.qpos.shape[0] == stack_scene.build_model().nq
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: FAIL with `ImportError: cannot import name 'HALF'`.

- [ ] **Step 3: Write the scene file**

`environments/assets/stack.xml`. The include brings in the Panda; blocks
use `contype`/`conaffinity` 3 so they collide with each other, the floor,
the hand capsule and the finger pads (Menagerie's arm links collide with
nothing).

```xml
<mujoco model="stack">
  <include file="panda/mjx_panda.xml"/>

  <option timestep="0.005" iterations="5" ls_iterations="8"
    integrator="implicitfast">
    <flag eulerdamp="disable"/>
  </option>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <asset>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.55 0.55 0.5"
      rgb2="0.5 0.5 0.45" width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="8 8"/>
  </asset>

  <worldbody>
    <light pos="0.5 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor"
      contype="1" conaffinity="1"/>
    <camera name="workspace" pos="1.05 0 0.55" fovy="50"
      xyaxes="0 1 0 -0.7071 0 0.7071"/>
    <body name="block_red" pos="1.5 -0.3 0.02">
      <freejoint name="block_red"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.85 0.1 0.1 1" mass="0.05"
        contype="3" conaffinity="3" condim="3" friction="1 0.03 0.003"
        solref="0.01 1"/>
    </body>
    <body name="block_green" pos="1.5 -0.1 0.02">
      <freejoint name="block_green"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.1 0.7 0.2 1" mass="0.05"
        contype="3" conaffinity="3" condim="3" friction="1 0.03 0.003"
        solref="0.01 1"/>
    </body>
    <body name="block_blue" pos="1.5 0.1 0.02">
      <freejoint name="block_blue"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.1 0.3 0.9 1" mass="0.05"
        contype="3" conaffinity="3" condim="3" friction="1 0.03 0.003"
        solref="0.01 1"/>
    </body>
    <body name="block_yellow" pos="1.5 0.3 0.02">
      <freejoint name="block_yellow"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.95 0.8 0.1 1" mass="0.05"
        contype="3" conaffinity="3" condim="3" friction="1 0.03 0.003"
        solref="0.01 1"/>
    </body>
  </worldbody>
</mujoco>
```

- [ ] **Step 4: Write the scene module**

Replace `environments/stack_scene.py` with:

```python
"""The MuJoCo scene for block stacking: a Franka Panda, four blocks and two
cameras.

The Panda is Menagerie's MJX model (assets/panda/), so the CPU simulator and
MJX simulate the same robot. Blocks not in the current episode are parked
behind the workspace camera, out of every view.
"""

import os

# EGL renders without a display, as DreamerV3's environment subprocesses
# need; set before mujoco is imported.
os.environ.setdefault('MUJOCO_GL', 'egl')

import pathlib  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parent / 'assets'
COLORS = ('red', 'green', 'blue', 'yellow')
HALF = 0.02  # Half the block edge.
SUBSTEPS = 10  # Physics steps per control step.
CONTROL_DT = 0.05
HOME_ARM = np.array([0, 0.3, 0, -1.57079, 0, 2.0, -0.7853])
OPEN, CLOSED = 0.04, 0.0  # Finger targets.
OPEN_WIDTH = 0.07  # Finger gap above which the gripper counts as open.


def build_model():
  """The scene with the wrist camera added to the Panda's hand."""
  spec = mujoco.MjSpec.from_file(str(ASSETS / 'stack.xml'))
  # The included Panda names its meshes relative to its own directory.
  spec.meshdir = str(ASSETS / 'panda' / 'assets')
  # In the hand frame z points from the wrist to the fingertips; turning the
  # camera half a turn about x makes it look along z, past the fingers.
  spec.body('hand').add_camera(
      name='wrist', pos=[0.06, 0, 0.0], quat=[0, 1, 0, 0], fovy=75)
  return spec.compile()


class Scene:
  """The model and data, with the block and gripper queries the task uses."""

  def __init__(self, size=(64, 64)):
    self.model = m = build_model()
    self.data = mujoco.MjData(m)
    self.size = tuple(size)
    self.renderer = None
    self.site = m.site('gripper').id
    self.bodies = [m.body(f'block_{c}').id for c in COLORS]
    joints = [m.joint(f'block_{c}').id for c in COLORS]
    self.qpos_adr = [int(m.jnt_qposadr[j]) for j in joints]
    self.qvel_adr = [int(m.jnt_dofadr[j]) for j in joints]
    self.color_of = {b: c for c, b in enumerate(self.bodies)}
    self.fingers = (m.body('left_finger').id, m.body('right_finger').id)
    self.reset_arm()

  def reset_arm(self):
    """Arm at home with the gripper open, every block parked."""
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[:7] = HOME_ARM
    d.qpos[7:9] = OPEN
    d.ctrl[:7] = HOME_ARM
    d.ctrl[7] = OPEN
    for c in range(len(COLORS)):
      self.park_block(c)
    self.forward()

  def set_block(self, c, pos, yaw):
    """Put block c at `pos` turned by `yaw` about z, at rest."""
    a, v = self.qpos_adr[c], self.qvel_adr[c]
    self.data.qpos[a:a + 3] = pos
    self.data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    self.data.qvel[v:v + 6] = 0
    self.forward()

  def park_block(self, c):
    self.set_block(c, (1.5, -0.3 + 0.2 * c, HALF), 0.0)

  def forward(self):
    mujoco.mj_forward(self.model, self.data)

  def step_physics(self, n):
    mujoco.mj_step(self.model, self.data, nstep=n)

  def positions(self):
    return self.data.xpos[self.bodies].copy()

  def quats(self):
    return self.data.xquat[self.bodies].copy()

  def velocities(self):
    """Linear velocities of the blocks, in the world frame."""
    return np.array([self.data.qvel[v:v + 3] for v in self.qvel_adr])

  def held(self):
    """Colours of the blocks touching both fingers."""
    touching = {}
    for con in self.data.contact[:self.data.ncon]:
      b1 = self.model.geom_bodyid[con.geom1]
      b2 = self.model.geom_bodyid[con.geom2]
      for block, other in ((b1, b2), (b2, b1)):
        if block in self.color_of and other in self.fingers:
          touching.setdefault(self.color_of[block], set()).add(other)
    return {c for c, fingers in touching.items() if len(fingers) == 2}

  def gripper_opening(self):
    return float(self.data.qpos[7] + self.data.qpos[8])
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: `7 passed`. If `test_mjx_loads_the_same_model` fails on an
unsupported feature, read the error: MJX rejects some geom pairs (for
example box-mesh); change the offending collision setting in `stack.xml`,
not the Panda copy, and rerun.

- [ ] **Step 6: Commit**

```bash
git add environments/assets/stack.xml environments/stack_scene.py tests/test_stack_scene.py
git commit -m "Add the block-stacking scene: Panda, floor, four blocks"
```

---

### Task 3: End-effector controller

**Files:**
- Create: `environments/stack_control.py`
- Test: `tests/test_stack_control.py`

**Interfaces:**
- Consumes: `Scene` and constants from Task 2.
- Produces, in `environments/stack_control.py`:
  - constants `LO = np.array([0.30, -0.22, 0.015])`,
    `HI = np.array([0.70, 0.22, 0.35])`, `MAX_STEP = 0.02`,
    `MAX_YAW = 0.2`, `YAW_LIMIT = np.pi / 2`
  - `class Controller(scene)` with `reset()`, `ee_pos() -> (3,)`,
    `ee_yaw() -> float`, `apply(action)` (one control step: sets the
    targets and runs `SUBSTEPS` physics steps).

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_control.py`:

```python
import numpy as np
import pytest

from environments.stack_control import HI, LO, MAX_STEP, Controller
from environments.stack_scene import HALF, Scene


@pytest.fixture
def ctrl():
  return Controller(Scene())


def settle(ctrl, steps=20):
  """Hold still until the arm reaches its (clipped) target."""
  for _ in range(steps):
    ctrl.apply(np.array([0, 0, 0, 0, -1.0]))


def move_to(ctrl, pos, grip, steps=60):
  """Drive the end effector towards `pos` with the gripper command `grip`."""
  for _ in range(steps):
    delta = (np.asarray(pos) - ctrl.target) / MAX_STEP
    action = np.concatenate([np.clip(delta, -1, 1), [0.0, grip]])
    ctrl.apply(action)


def test_starting_target_is_inside_the_bounds(ctrl):
  assert (ctrl.target >= LO).all() and (ctrl.target <= HI).all()


def test_moves_along_x(ctrl):
  settle(ctrl)
  start = ctrl.ee_pos()
  for _ in range(5):
    ctrl.apply(np.array([1.0, 0, 0, 0, -1]))
  for _ in range(10):
    ctrl.apply(np.array([0.0, 0, 0, 0, -1]))
  moved = ctrl.ee_pos() - start
  assert moved[0] == pytest.approx(0.10, abs=0.01)
  assert abs(moved[1]) < 0.01 and abs(moved[2]) < 0.01


def test_turns_yaw(ctrl):
  settle(ctrl)
  for _ in range(4):
    ctrl.apply(np.array([0, 0, 0, 1.0, -1]))
  for _ in range(10):
    ctrl.apply(np.array([0, 0, 0, 0.0, -1]))
  assert ctrl.ee_yaw() == pytest.approx(0.8, abs=0.05)


def test_random_actions_stay_bounded(ctrl):
  rng = np.random.default_rng(0)
  for _ in range(400):
    ctrl.apply(rng.uniform(-1, 1, 5))
    pos = ctrl.ee_pos()
    assert np.isfinite(ctrl.scene.data.qpos).all()
    assert (pos > LO - 0.03).all() and (pos < HI + 0.03).all()


def test_grasps_and_lifts_a_block(ctrl):
  scene = ctrl.scene
  scene.set_block(0, (0.5, 0.0, HALF), 0.0)
  move_to(ctrl, (0.5, 0.0, 0.12), -1)
  move_to(ctrl, (0.5, 0.0, HALF), -1)
  move_to(ctrl, (0.5, 0.0, HALF), 1, steps=15)
  move_to(ctrl, (0.5, 0.0, 0.15), 1)
  assert scene.positions()[0][2] > 0.10
  assert scene.held() == {0}
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_control.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack_control'`.

- [ ] **Step 3: Write the controller**

`environments/stack_control.py`:

```python
"""End-effector control for the Panda: 5-D actions to joint targets.

An action moves a target for the gripper site by up to MAX_STEP per control
step, turns its yaw by up to MAX_YAW, and opens or closes the gripper. The
gripper always points down. Damped least-squares IK on a scratch copy of the
data turns the target into joint position targets for the arm's actuators.
"""

import mujoco
import numpy as np

from environments.stack_scene import CLOSED, OPEN, SUBSTEPS

LO = np.array([0.30, -0.22, 0.015])  # Gripper site bounds, metres.
HI = np.array([0.70, 0.22, 0.35])
MAX_STEP = 0.02
MAX_YAW = 0.2
YAW_LIMIT = np.pi / 2


def rot_z(angle):
  c, s = np.cos(angle), np.sin(angle)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Controller:

  def __init__(self, scene, iters=5, damping=0.05):
    self.scene = scene
    self.iters, self.damping = iters, damping
    self.scratch = mujoco.MjData(scene.model)
    self.ranges = scene.model.jnt_range[:7].copy()
    scene.reset_arm()
    # Yaw is measured from the gripper's orientation at the home pose.
    self.home_mat = scene.data.site_xmat[scene.site].reshape(3, 3).copy()
    self.reset()

  def reset(self):
    """Targets at the current gripper pose, moved inside the bounds."""
    self.target = np.clip(self.ee_pos(), LO, HI)
    self.yaw = float(np.clip(self.ee_yaw(), -YAW_LIMIT, YAW_LIMIT))

  def ee_pos(self):
    return self.scene.data.site_xpos[self.scene.site].copy()

  def ee_yaw(self):
    mat = self.scene.data.site_xmat[self.scene.site].reshape(3, 3)
    rel = mat @ self.home_mat.T
    return float(np.arctan2(rel[1, 0], rel[0, 0]))

  def apply(self, action):
    """One control step."""
    a = np.clip(np.asarray(action, np.float64), -1, 1)
    self.target = np.clip(self.target + MAX_STEP * a[:3], LO, HI)
    self.yaw = float(np.clip(self.yaw + MAX_YAW * a[3], -YAW_LIMIT, YAW_LIMIT))
    data = self.scene.data
    data.ctrl[:7] = self.solve(self.target, rot_z(self.yaw) @ self.home_mat)
    data.ctrl[7] = CLOSED if a[4] > 0 else OPEN
    self.scene.step_physics(SUBSTEPS)

  def solve(self, pos, mat):
    """Arm joint angles that put the gripper site at `pos` with
    orientation `mat`."""
    m, s, site = self.scene.model, self.scratch, self.scene.site
    s.qpos[:] = self.scene.data.qpos
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    quat, rot_err = np.zeros(4), np.zeros(3)
    for _ in range(self.iters):
      mujoco.mj_kinematics(m, s)
      mujoco.mj_comPos(m, s)
      mujoco.mj_jacSite(m, s, jacp, jacr, site)
      jac = np.vstack([jacp[:, :7], jacr[:, :7]])
      cur = s.site_xmat[site].reshape(3, 3)
      mujoco.mju_mat2Quat(quat, (mat @ cur.T).flatten())
      mujoco.mju_quat2Vel(rot_err, quat, 1.0)
      err = np.concatenate([pos - s.site_xpos[site], rot_err])
      lhs = jac @ jac.T + self.damping ** 2 * np.eye(6)
      dq = jac.T @ np.linalg.solve(lhs, err)
      s.qpos[:7] = np.clip(s.qpos[:7] + dq, self.ranges[:, 0], self.ranges[:, 1])
    return s.qpos[:7].copy()
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_control.py -q`
Expected: `5 passed`. If `test_grasps_and_lifts_a_block` fails because the
block slips, check the grasp height first: print
`ctrl.ee_pos()[2]` and the block's z after the descent. The fingertip pads
sit about 1 cm above the gripper site; lowering `LO[2]` or the descent
target by up to 1 cm is the expected fix, not changing friction.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_control.py tests/test_stack_control.py
git commit -m "Add end-effector control for the Panda by damped least-squares IK"
```

---

### Task 4: Cameras

**Files:**
- Modify: `environments/stack_scene.py`
- Test: `tests/test_stack_render.py`

**Interfaces:**
- Consumes: `Scene` (Task 2), `Controller` (Task 3, in the test only).
- Produces: `Scene.render() -> {'workspace': uint8 (H, W, 3), 'wrist': uint8 (H, W, 3)}`
  with `(H, W) = scene.size`.

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_render.py`:

```python
import subprocess
import sys

import numpy as np

from environments.stack_scene import HALF, Scene
from conftest import REPO


def red_pixels(image):
  r, g, b = (image[..., i].astype(int) for i in range(3))
  return int(((r > 150) & (g < 80) & (b < 80)).sum())


def test_both_cameras_render():
  scene = Scene()
  images = scene.render()
  assert set(images) == {'workspace', 'wrist'}
  for image in images.values():
    assert image.shape == (64, 64, 3) and image.dtype == np.uint8
    assert image.std() > 5


def test_workspace_camera_sees_a_block():
  scene = Scene()
  scene.set_block(0, (0.5, 0.0, HALF), 0.0)
  assert red_pixels(scene.render()['workspace']) >= 4


def test_wrist_camera_sees_a_block_below():
  from environments.stack_control import Controller
  scene = Scene()
  ctrl = Controller(scene)
  scene.set_block(0, (0.5, 0.0, HALF), 0.0)
  for _ in range(60):
    delta = (np.array([0.5, 0.0, 0.15]) - ctrl.target) / 0.02
    ctrl.apply(np.concatenate([np.clip(delta, -1, 1), [0, -1]]))
  assert red_pixels(scene.render()['wrist']) >= 20


def test_renders_in_fresh_headless_process():
  code = ('from environments.stack_scene import Scene; '
          'print(Scene().render()["workspace"].shape)')
  env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': str(REPO)}
  out = subprocess.run([sys.executable, '-c', code], env=env, cwd=REPO,
                       capture_output=True, text=True, timeout=120)
  assert out.returncode == 0, out.stderr
  assert '(64, 64, 3)' in out.stdout
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_render.py -q`
Expected: FAIL with `AttributeError: 'Scene' object has no attribute 'render'`.

- [ ] **Step 3: Add rendering to `Scene`**

Add this method to `Scene` in `environments/stack_scene.py`:

```python
  def render(self):
    """Both camera images, uint8 RGB at the scene's size."""
    if self.renderer is None:
      self.renderer = mujoco.Renderer(self.model, *self.size)
    images = {}
    for camera in ('workspace', 'wrist'):
      self.renderer.update_scene(self.data, camera=camera)
      images[camera] = self.renderer.render().copy()
    return images
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_render.py -q`
Expected: `4 passed`. If the wrist test fails, save the image
(`PIL.Image.fromarray(img).resize((256, 256)).save('/tmp/wrist.png')`) and
look at it: adjust the camera's `pos`/`quat` in `build_model()` until the
fingertips sit at the image edge and the block is in the centre.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_scene.py tests/test_stack_render.py
git commit -m "Render the workspace and wrist cameras for block stacking"
```

---

### Task 5: Task logic

**Files:**
- Create: `environments/stack_task.py`
- Test: `tests/test_stack_task.py`

**Interfaces:**
- Consumes: `HALF` and `COLORS` from `environments.stack_scene`.
- Produces, in `environments/stack_task.py`:
  - constants `NONE = 4`, `LIMITS = {2: 400, 3: 600, 4: 800}`,
    `HOLD_STEPS = 20`, `XY_TOL = 0.015`, `Z_TOL = 0.006`,
    `CENTER = (0.5, 0.0)`, `EXTENT = 0.15`, `MIN_SEP = 0.07`
  - `Start` (namedtuple: `n`, `goal` tuple of colour indices bottom to top,
    `positions` dict colour -> (3,), `yaws` dict colour -> float)
  - `sample_start(rng, n=None, prestack=0.3) -> Start`
  - `tower_prefix(goal, positions, held) -> int` where `positions` indexes
    by colour (array (4, 3) or dict)
  - `class Progress(n)` with `reset(k)` and
    `update(k, gripper_open) -> (reward, success)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_task.py`:

```python
import numpy as np
import pytest

from environments.stack_scene import HALF
from environments.stack_task import (
    CENTER, EXTENT, HOLD_STEPS, MIN_SEP, Progress, sample_start, tower_prefix)


def tower(goal, xy=(0.5, 0.0)):
  """Positions of a perfect tower in goal order."""
  pos = np.full((4, 3), 9.0)
  for level, c in enumerate(goal):
    pos[c] = (xy[0], xy[1], HALF + 2 * HALF * level)
  return pos


def test_prefix_of_a_full_tower():
  assert tower_prefix((2, 0, 1), tower((2, 0, 1)), set()) == 3


def test_prefix_stops_at_the_wrong_order():
  pos = tower((2, 1, 0))
  assert tower_prefix((2, 0, 1), pos, set()) == 1


def test_prefix_is_zero_when_bottom_block_is_lifted():
  pos = tower((2, 0))
  pos[2][2] += 0.05
  pos[0][2] += 0.05
  assert tower_prefix((2, 0), pos, set()) == 0


def test_prefix_ignores_blocks_that_are_held():
  assert tower_prefix((2, 0, 1), tower((2, 0, 1)), {1}) == 2


def test_prefix_needs_alignment():
  pos = tower((2, 0))
  pos[0][0] += 0.02
  assert tower_prefix((2, 0), pos, set()) == 1


def test_progress_rewards_changes_in_prefix():
  p = Progress(3)
  p.reset(1)
  assert p.update(2, False) == (1.0, False)
  assert p.update(2, False) == (0.0, False)
  assert p.update(1, False) == (-1.0, False)


def test_success_needs_the_hold():
  p = Progress(2)
  p.reset(1)
  reward, success = p.update(2, True)
  assert (reward, success) == (1.0, False)
  for _ in range(HOLD_STEPS - 1):
    assert p.update(2, True) == (0.0, False)
  assert p.update(2, True) == (1.0, True)


def test_timeout_before_hold_is_not_success():
  p = Progress(2)
  p.reset(1)
  p.update(2, True)
  results = [p.update(2, True) for _ in range(HOLD_STEPS - 2)]
  assert not any(success for _, success in results)


def test_hold_restarts_when_gripper_closes():
  p = Progress(2)
  p.reset(2)
  for _ in range(HOLD_STEPS - 1):
    p.update(2, True)
  p.update(2, False)
  assert p.update(2, True) == (0.0, False)


def test_starts_are_separated_and_in_the_workspace():
  rng = np.random.default_rng(0)
  for _ in range(200):
    start = sample_start(rng, prestack=0.0)
    assert 2 <= start.n <= 4 and len(start.goal) == start.n
    xys = [start.positions[c][:2] for c in start.goal]
    for xy in xys:
      assert abs(xy[0] - CENTER[0]) <= EXTENT
      assert abs(xy[1] - CENTER[1]) <= EXTENT
    for i in range(len(xys)):
      for j in range(i):
        assert np.linalg.norm(xys[i] - xys[j]) >= MIN_SEP


def test_prestacked_starts_stack_two_or_three():
  rng = np.random.default_rng(1)
  heights = set()
  for _ in range(200):
    start = sample_start(rng, prestack=1.0)
    zs = sorted(start.positions[c][2] for c in start.goal)
    stacked = sum(z > HALF + 1e-6 for z in zs) + 1
    heights.add(stacked)
  assert heights <= {2, 3} and heights


def test_fixed_n():
  rng = np.random.default_rng(2)
  assert {sample_start(rng, n=3).n for _ in range(20)} == {3}
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_task.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack_task'`.

- [ ] **Step 3: Write the task logic**

`environments/stack_task.py`:

```python
"""Block-stacking task logic: start states, tower progress, reward and
success. Pure NumPy, no simulator.

The tower prefix k counts goal blocks correctly in place from the bottom:
the first rests on the table, each next one on the one below it, and none
is held. The reward is the change in k each step, plus 1 on success, which
is k = N with the gripper open for HOLD_STEPS control steps.
"""

import collections

import numpy as np

from environments.stack_scene import COLORS, HALF

NONE = len(COLORS)  # Goal padding.
LIMITS = {2: 400, 3: 600, 4: 800}  # Control steps per episode, by N.
HOLD_STEPS = 20  # 1 s at 20 Hz.
XY_TOL = 0.015
Z_TOL = 0.006
CENTER = (0.5, 0.0)
EXTENT = 0.15  # Half the workspace side.
MIN_SEP = 0.07  # Between block centres at the start.

Start = collections.namedtuple('Start', 'n goal positions yaws')


def tower_prefix(goal, positions, held):
  """How many goal blocks, from the bottom, are in place."""
  k = 0
  for level, c in enumerate(goal):
    if c in held:
      break
    pos = np.asarray(positions[c])
    if level == 0:
      ok = abs(pos[2] - HALF) < Z_TOL
    else:
      below = np.asarray(positions[goal[level - 1]])
      ok = (np.linalg.norm(pos[:2] - below[:2]) < XY_TOL and
            abs(pos[2] - below[2] - 2 * HALF) < Z_TOL)
    if not ok:
      break
    k += 1
  return k


class Progress:
  """Reward and success from the tower prefix, step by step."""

  def __init__(self, n):
    self.n = n

  def reset(self, k):
    self.k, self.count = k, 0

  def update(self, k, gripper_open):
    reward = float(k - self.k)
    self.k = k
    self.count = self.count + 1 if (k == self.n and gripper_open) else 0
    success = self.count > HOLD_STEPS
    return reward + float(success), success


def sample_start(rng, n=None, prestack=0.3):
  """A random goal and block layout, sometimes with a stack already built."""
  n = int(n or rng.integers(2, 5))
  goal = tuple(int(c) for c in rng.permutation(len(COLORS))[:n])
  positions, yaws, xys = {}, {}, []
  loose = list(goal)
  if n >= 2 and rng.random() < prestack:
    height = int(rng.integers(2, min(3, n) + 1))
    members = [int(c) for c in rng.permutation(goal)[:height]]
    xy = free_xy(rng, xys)
    yaw = float(rng.uniform(-np.pi / 4, np.pi / 4))
    for level, c in enumerate(members):
      positions[c] = np.array([xy[0], xy[1], HALF + 2 * HALF * level])
      yaws[c] = yaw
      loose.remove(c)
  for c in loose:
    xy = free_xy(rng, xys)
    positions[c] = np.array([xy[0], xy[1], HALF])
    yaws[c] = float(rng.uniform(-np.pi / 4, np.pi / 4))
  return Start(n, goal, positions, yaws)


def free_xy(rng, taken):
  """A workspace point at least MIN_SEP from every point in `taken`, which
  it is appended to."""
  while True:
    xy = np.array(CENTER) + rng.uniform(-EXTENT, EXTENT, 2)
    if all(np.linalg.norm(xy - t) >= MIN_SEP for t in taken):
      taken.append(xy)
      return xy
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_task.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_task.py tests/test_stack_task.py
git commit -m "Add block-stacking task logic: starts, tower prefix, reward"
```

---

### Task 6: The `stack` environment

**Files:**
- Create: `environments/stack.py`
- Modify: `environments/__init__.py` (add to `REGISTRY`)
- Test: `tests/test_stack_env.py`

**Interfaces:**
- Consumes: `Scene` (Tasks 2, 4), `Controller` (Task 3), everything in
  `stack_task` (Task 5).
- Produces: `environments.make('stack', **options)` returning
  `Stack(task, size=(64, 64), n=0, prestack=0.3, seed=None)`, where `n=0`
  samples N per episode. Observation keys `workspace`, `wrist` (uint8
  64x64x3), `arm` (float32 (19,)), `goal` (int32 (4,), 5 classes),
  `reward`, `is_first`, `is_last`, `is_terminal`, `log/prefix`, `log/n`,
  `log/success`, `log/sim_steps` (scalars). Action keys `reset`, `action`
  (float32 (5,) in [-1, 1]). Method `privileged() -> dict(blocks=float32
  (4, 11), held=int)`. Attributes `scene`, `control`, `goal`, `n`. Later
  plans' option environment wraps this class.

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_env.py`:

```python
import numpy as np
import pytest

import environments
from environments.stack_scene import HALF
from environments.stack_task import HOLD_STEPS, LIMITS

RESET = {'reset': True, 'action': np.zeros(5, np.float32)}


def act(a):
  return {'reset': False, 'action': np.asarray(a, np.float32)}


@pytest.fixture
def env():
  return environments.make('stack', seed=0)


def check_obs(env, obs):
  for key, space in env.obs_space.items():
    value = np.asarray(obs[key])
    assert value.shape == space.shape, key
    assert value.dtype == space.dtype, key
    if key.startswith('log/'):
      assert value.ndim == 0, key
  assert np.isfinite(obs['arm']).all()


def build_tower(env):
  """Teleport the episode's blocks into the goal tower and settle."""
  for level, c in enumerate(env.goal):
    env.scene.set_block(c, (0.5, 0.15, HALF + 2 * HALF * level), 0.0)
  env.scene.step_physics(40)


def test_reset_observation(env):
  obs = env.step(RESET)
  check_obs(env, obs)
  assert obs['is_first'] and not obs['is_last']
  n = int(obs['log/n'])
  assert (obs['goal'][:n] < 4).all() and (obs['goal'][n:] == 4).all()


def test_step_observation(env):
  env.step(RESET)
  obs = env.step(act([0.5, 0, 0, 0, -1]))
  check_obs(env, obs)
  assert not obs['is_first']
  assert int(obs['log/sim_steps']) == 1


def test_reset_starts_at_rest():
  for seed in range(30):
    env = environments.make('stack', seed=seed, prestack=0.5)
    env.step(RESET)
    blocks = env.privileged()['blocks']
    present = blocks[:, 0] > 0
    assert np.abs(blocks[present, 8:11]).max() < 0.02, seed


def test_privileged_marks_present_blocks(env):
  obs = env.step(RESET)
  blocks = env.privileged()['blocks']
  assert blocks.shape == (4, 11)
  present = set(np.flatnonzero(blocks[:, 0]))
  assert present == set(env.goal)
  assert int(obs['log/n']) == len(env.goal)


def test_building_the_tower_succeeds():
  env = environments.make('stack', seed=3, n=2, prestack=0.0)
  env.step(RESET)
  build_tower(env)
  total, obs = 0.0, None
  for _ in range(HOLD_STEPS + 5):
    obs = env.step(act([0, 0, 0, 0, -1]))
    total += float(obs['reward'])
    if obs['is_last']:
      break
  assert obs['is_last'] and obs['is_terminal']
  assert int(obs['log/success']) == 1
  assert total == pytest.approx(2.0)  # Prefix 1 -> 2, plus success.


def test_toppling_gives_negative_reward():
  env = environments.make('stack', seed=4, n=3, prestack=0.0)
  env.step(RESET)
  build_tower(env)
  env.step(act([0, 0, 0, 0, -1]))
  top = env.goal[-1]
  env.scene.set_block(top, (0.6, -0.1, HALF), 0.0)
  obs = env.step(act([0, 0, 0, 0, -1]))
  assert float(obs['reward']) == pytest.approx(-1.0)
  assert not obs['is_last']


def test_time_limit_ends_episode():
  env = environments.make('stack', seed=5, n=2, prestack=0.0)
  env.step(RESET)
  obs = None
  for _ in range(LIMITS[2]):
    obs = env.step(act([0, 0, 0, 0, -1]))
  assert obs['is_last'] and not obs['is_terminal']
  assert env.step(act([0, 0, 0, 0, -1]))['is_first']


def test_same_seed_same_start():
  a = environments.make('stack', seed=7)
  b = environments.make('stack', seed=7)
  a.step(RESET)
  b.step(RESET)
  assert a.goal == b.goal
  np.testing.assert_allclose(
      a.privileged()['blocks'], b.privileged()['blocks'], atol=1e-6)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_env.py -q`
Expected: FAIL with `KeyError: "Unknown custom environment 'stack'..."`.

- [ ] **Step 3: Write the environment**

`environments/stack.py`:

```python
"""Block stacking with a Franka Panda, acting through end-effector deltas.

Each episode samples N blocks (2-4) and a goal order; the agent sees a
workspace camera, a wrist camera, the arm's state and the goal, and moves the
gripper by up to 2 cm and 0.2 rad of yaw per 20 Hz control step. Scalar
diagnostics are log/ keys; the full block state comes from privileged().
"""

import elements
import embodied
import numpy as np

from environments.stack_control import Controller
from environments.stack_scene import COLORS, OPEN_WIDTH, Scene
from environments.stack_task import (
    LIMITS, NONE, Progress, sample_start, tower_prefix)

ARM_DIM = 19  # 7 joint angles, 7 velocities, gripper xyz, yaw, opening.
SETTLE = 60  # Physics steps (0.3 s) to let a new layout come to rest.


class Stack(embodied.Env):

  def __init__(self, task, size=(64, 64), n=0, prestack=0.3, seed=None):
    del task
    self.size = tuple(size)
    self.fixed_n = int(n) or None
    self.prestack = prestack
    self.rng = np.random.default_rng(seed)
    self.scene = Scene(self.size)
    self.control = Controller(self.scene)
    self.sim_steps = 0
    self.done = True

  @property
  def obs_space(self):
    image = elements.Space(np.uint8, self.size + (3,))
    return {
        'workspace': image,
        'wrist': image,
        'arm': elements.Space(np.float32, (ARM_DIM,)),
        'goal': elements.Space(np.int32, (len(COLORS),), 0, NONE + 1),
        'reward': elements.Space(np.float32),
        'is_first': elements.Space(bool),
        'is_last': elements.Space(bool),
        'is_terminal': elements.Space(bool),
        'log/prefix': elements.Space(np.int32),
        'log/n': elements.Space(np.int32),
        'log/success': elements.Space(np.int32),
        'log/sim_steps': elements.Space(np.int32),
    }

  @property
  def act_space(self):
    return {
        'reset': elements.Space(bool),
        'action': elements.Space(np.float32, (5,), -1, 1),
    }

  def step(self, action):
    if action['reset'] or self.done:
      return self.reset()
    self.control.apply(action['action'])
    self.sim_steps += 1
    self.steps += 1
    k = self.prefix()
    opened = self.scene.gripper_opening() > OPEN_WIDTH
    reward, success = self.progress.update(k, opened)
    self.done = success or self.steps >= LIMITS[self.n]
    return self.obs(reward, k, success, is_last=self.done,
                    is_terminal=success)

  def reset(self):
    start = sample_start(self.rng, self.fixed_n, self.prestack)
    self.n, self.goal = start.n, start.goal
    self.scene.reset_arm()
    for c in self.goal:
      self.scene.set_block(c, start.positions[c], start.yaws[c])
    self.scene.step_physics(SETTLE)
    self.control.reset()
    self.steps = 0
    self.done = False
    k = self.prefix()
    self.progress = Progress(self.n)
    self.progress.reset(k)
    return self.obs(0.0, k, False, is_first=True)

  def prefix(self):
    return tower_prefix(self.goal, self.scene.positions(), self.scene.held())

  def privileged(self):
    """The simulator's block state, which the agent does not observe."""
    blocks = np.zeros((len(COLORS), 11), np.float32)
    pos, quat, vel = (self.scene.positions(), self.scene.quats(),
                      self.scene.velocities())
    for c in self.goal:
      blocks[c] = np.concatenate([[1.0], pos[c], quat[c], vel[c]])
    held = self.scene.held()
    return dict(blocks=blocks, held=min(held) if held else -1)

  def obs(self, reward, k, success, is_first=False, is_last=False,
          is_terminal=False):
    d = self.scene.data
    arm = np.concatenate([
        d.qpos[:7], d.qvel[:7], self.control.ee_pos(),
        [self.control.ee_yaw(), self.scene.gripper_opening()]])
    goal = np.full(len(COLORS), NONE, np.int32)
    goal[:self.n] = self.goal
    return dict(
        **self.scene.render(),
        arm=arm.astype(np.float32),
        goal=goal,
        reward=np.float32(reward),
        is_first=is_first, is_last=is_last, is_terminal=is_terminal,
        **{'log/prefix': np.int32(k), 'log/n': np.int32(self.n),
           'log/success': np.int32(success),
           'log/sim_steps': np.int32(self.sim_steps)})
```

Add to `REGISTRY` in `environments/__init__.py`:

```python
REGISTRY = {
    'catch': 'environments.catch:Catch',
    'stack': 'environments.stack:Stack',
    'tetris': 'environments.tetris:Tetris',
}
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_env.py -q`
Expected: `8 passed`. The bottom goal block starts on the table, so k is 1
at reset; `test_building_the_tower_succeeds` expects +1 for k going from 1
to 2 and +1 for success. If it fails, print `env.prefix()` and
`env.scene.held()` after `build_tower`.

- [ ] **Step 5: Run the whole suite**

Run: `$PY -m pytest tests -q`
Expected: all tests pass (36).

- [ ] **Step 6: Commit**

```bash
git add environments/stack.py environments/__init__.py tests/test_stack_env.py
git commit -m "Add the stack environment: raw end-effector block stacking"
```

---

### Task 7: Playing `stack` by hand

**Files:**
- Modify: `environments/play.py`

**Interfaces:**
- Consumes: `environments.make('stack')` (Task 6).
- Produces: `python environments/play.py stack` showing both cameras.
  `KEYS` values may be an `int` (discrete action, as before) or a
  `(dimension, sign)` pair for continuous environments; `('grip', None)`
  toggles the gripper.

- [ ] **Step 1: Generalise `render` to every camera in the observation**

In `environments/play.py`, replace the start of `render` so it draws every
uint8 HxWx3 observation, in sorted key order:

```python
def render(env, obs, scale, screen=True):
  """Every camera image in the observation, after the emulator screen for
  Atari environments when `screen` is set."""
  images = [obs[k] for k in sorted(obs)
            if getattr(obs[k], 'dtype', None) == np.uint8
            and np.ndim(obs[k]) == 3]
  images = [np.repeat(i, 3, -1) if i.shape[-1] == 1 else i for i in images]
  panels = [Image.fromarray(i).resize(
      (i.shape[1] * scale, i.shape[0] * scale), Image.NEAREST)
      for i in images]
```

and keep the rest of the function (the emulator screen and canvas) as it is.

- [ ] **Step 2: Add the stack keys and continuous actions**

Add to `KEYS` and `HELP`:

```python
    'stack': {
        'w': (0, 1), 's': (0, -1), 'a': (1, 1), 'd': (1, -1),
        'r': (2, 1), 'f': (2, -1), 'q': (3, 1), 'e': (3, -1),
        'space': ('grip', None)},
```

```python
    'stack': 'W/S move x, A/D move y, R/F up/down, Q/E turn, '
             'Space opens or closes the gripper',
```

In `Player`, track the gripper and build a continuous action when the key's
value is a pair. Replace `reset` and `press`:

```python
  def reset(self):
    self.score, self.steps, self.grip = 0.0, 0, -1.0
    obs = self.env.step({'reset': True, 'action': self.zero_action()})
    self.show(obs, 'new episode')

  def zero_action(self):
    space = self.env.act_space['action']
    return np.zeros(space.shape, space.dtype) if space.shape else 0

  def press(self, event):
    if event.keysym == 'Escape':
      self.root.destroy()
    elif event.keysym in ('n', 'N'):
      self.reset()
    elif event.keysym in self.keys and not self.done:
      key = self.keys[event.keysym]
      if isinstance(key, tuple):
        action = self.zero_action()
        if key[0] == 'grip':
          self.grip = -self.grip
        else:
          action[key[0]] = key[1]
        action[-1] = self.grip
      else:
        action = key
      obs = self.env.step({'reset': False, 'action': action})
      self.score += float(obs['reward'])
      self.steps += 1
      self.show(obs, f'action {np.round(action, 2)} ({event.keysym})')
```

- [ ] **Step 3: Check the existing environments still play**

Run: `$PY environments/play.py catch` and press Left/Right a few times, then
Escape. Run `$PY environments/play.py tetris` and press a digit. Both must
behave as before.

- [ ] **Step 4: Play `stack` by hand**

Run: `$PY environments/play.py stack --scale 6`
Check, and say in the commit message what was seen: both camera panels
show; W/S/A/D/R/F move the gripper the right way in the workspace camera;
Space closes the gripper; you can pick up one block and set it on another,
and the status line's reward shows +1 when the tower grows.

- [ ] **Step 5: Commit**

```bash
git add environments/play.py
git commit -m "Play the stack environment by hand with both cameras"
```

---

### Task 8: DreamerV3 trains on `stack`

**Files:**
- Modify: `environments/README.md`

**Interfaces:**
- Consumes: `custom_stack` (Task 6), `train.py`.
- Produces: a confirmed DreamerV3 run on `custom_stack` and its docs.

- [ ] **Step 1: Short CPU training run**

```bash
cd /home/max/PlanningWithWorldModels
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
scratch=/tmp/claude-1000/-home-max-PlanningWithWorldModels/stack_debug
rm -rf $scratch
$PY train.py --task custom_stack --configs debug --logdir $scratch \
  --run.steps 3500 2>&1 | tail -20
```

`run.steps` counts steps over all 4 environments, and an episode lasts up
to 800 control steps (N = 4), so 3500 steps ends at least one episode in
every environment.

Expected: the run finishes without a traceback, and `$scratch/metrics.jsonl`
has `episode/score` lines.

```bash
grep -c . $scratch/metrics.jsonl
grep -o '"epstats/log/[a-z_]*/avg"' $scratch/metrics.jsonl | sort -u
ls $scratch/scope | head
```

Expected: `epstats/log/n/avg`, `epstats/log/prefix/avg`,
`epstats/log/sim_steps/avg`, `epstats/log/success/avg` among the keys, and
policy videos for both cameras (`epstats-policy_workspace.mp4`,
`epstats-policy_wrist.mp4`) in `scope/`.

- [ ] **Step 2: Document the environment**

Add under "## Environments" in `environments/README.md`:

```markdown
- `stack`: a Franka Panda in MuJoCo stacks 2-4 coloured blocks in a goal
  order. The agent sees a workspace camera and a wrist camera (64x64 RGB),
  the arm's state and the goal, and moves the gripper by end-effector deltas
  at 20 Hz. Options: `n` fixes N (0 samples 2-4), `prestack` is the share of
  starts with a stack already built, `size` the camera resolution. Needs
  `mujoco` (see `constraints.txt` for installing into the dreamerv3 env) and
  renders with EGL. Block poses are not observed; `privileged()` returns
  them for evaluation.
```

and under "## Playing by hand":

```markdown
    python environments/play.py stack --scale 6
```

- [ ] **Step 3: Commit**

```bash
git add environments/README.md
git commit -m "Document the stack environment"
```

---

## Later plans

Each is written in full when the one before it is done, using what it showed.

- **Plan 2: Skills.** `skills/mjx_env.py` (pick and place on MJX, sharing
  `stack_scene.build_model()` and `stack_task.sample_start`),
  `skills/teacher.py` (Brax PPO state teachers with a terminate output),
  `skills/student.py` (CNN image students by DAgger on `Stack`),
  `skills/policy.py` (loading a student). Gates: teacher and student each at
  least 90% of 500 episodes on CPU MuJoCo. Installs `brax` with
  `-c constraints.txt`.
- **Plan 3: Options and training runs.** `environments/stack_options.py`
  (`custom_stack_options`: 9 options, validity masks in `privileged()`,
  10-step cost for invalid options, `log/duration`, summed reward, time-limit
  cut-off); the N = 2 pilot for both agents; main runs at the budget the
  pilot sets.
- **Plan 4: Planners and evaluation.** iCEM and sampled MuZero in
  `planning/search.py` for both action spaces; `eval_model.py` for custom
  environments, continuous actions and success by N; `planning/model_error.py`
  with pose probes; the results table.
