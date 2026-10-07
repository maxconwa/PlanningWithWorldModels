# Block Stacking Environment Implementation Plan (Stage 1 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A GPU-batched MuJoCo Panda block-stacking environment with raw
end-effector actions and two 64 x 64 cameras, registered as `custom_stack`,
tested, playable by hand, and trained on by DreamerV3 through `train.py`.

**Architecture:** Physics and rendering run on the GPU in MJX's Warp backend
(MuJoCo Warp physics, MuJoCo Warp batch renderer) as JAX functions over a
batch of worlds. Small modules in `environments/`: `stack_scene.py` builds
the MuJoCo model; `stack_sim.py` holds the batched physics and rendering;
`stack_task.py` the task logic and `stack_control.py` the IK controller, both
in JAX; `stack.py` joins them as `BatchStack` (all worlds) and `Stack` (the
`embodied.Env` DreamerV3 and `play.py` see). `batch_driver.py` lets
DreamerV3 step a whole batch at once; `train.py` swaps it in.

**Tech Stack:** Python 3.11, JAX 0.10.2, `mujoco` 3.15.0,
`mujoco-mjx[warp]` 3.15.0 (`warp-lang` 1.17.0), DreamerV3's
`embodied`/`elements`, pytest. RTX 5070 Ti (Blackwell).

**Spec:** `docs/superpowers/specs/2026-10-07-block-stacking-options-design.md`

## Global Constraints

- Python with 2-space indentation, single quotes, lines of about 80 columns,
  no type hints; short docstrings saying what a function is for; comments
  describe the code as it is (AGENTS.md "Code style").
- Do not edit `third_party/` (AGENTS.md "Layout").
- `PY=~/miniconda3/envs/dreamerv3/bin/python`. JAX stays at 0.10.2: every
  install uses `-c constraints.txt`.
- One GPU process at a time (AGENTS.md "Python environments"): check
  `nvidia-smi` is empty before running GPU tests or training.
- Physics and rendering always go through `environments/stack_sim.py` (MJX
  Warp backend). CPU MuJoCo appears only in tests, as a reference.
- DreamerV3 sets `jax_transfer_guard='disallow'`: the environment's step
  moves NumPy inputs with `jax.device_put` and results with
  `jax.device_get`, and indexes device arrays only inside jitted functions.
- Four colours, in this order everywhere: red, green, blue, yellow (indices
  0-3); `NONE = 4` pads the goal.
- Blocks are 4 cm cubes (`HALF = 0.02`).
- Physics timestep 0.005 s, 10 physics steps per control step, control at
  20 Hz.
- Raw action: 5 floats in [-1, 1]: dx, dy, dz (up to 0.02 m per step), dyaw
  (up to 0.2 rad per step), gripper (> 0 closes).
- Time limits in control steps: N = 2: 400, N = 3: 600, N = 4: 800.
- Success: tower prefix k = N with the gripper open, held for more than 20
  control steps (1 s). Reward: change in k each step, plus 1 on success.
- 30% of starts pre-stack 2 or 3 of the blocks.
- `log/` observation keys are scalars per world (DreamerV3's
  `embodied/run/train.py:42` asserts it). Arrays go through `privileged()`.

## Review Focus

1. **Blocks spawning inside each other or a stack collapsing at reset.**
   Expected: every episode starts with blocks at rest. Test: Task 6,
   `test_reset_starts_at_rest` over 64 worlds.
2. **Random actions driving the arm into the floor, out of reach, or into
   NaNs.** Expected: the gripper stays inside the bounds and observations
   stay finite. Test: Task 5, `test_random_actions_stay_bounded`.
3. **A built tower being knocked over.** Expected: k drops and that step's
   reward is negative; the episode goes on. Test: Task 6,
   `test_toppling_gives_negative_reward`.
4. **Worlds in one batch ending at different times.** Expected: a world that
   ends resets on its own next step, and the others carry on untouched.
   Test: Task 6, `test_worlds_reset_independently`.
5. **Transfer guard under DreamerV3.** Expected: building and stepping the
   environment with `jax_transfer_guard='disallow'` works. Test: Task 6,
   `test_runs_under_transfer_guard`.

## File Structure

- Create `constraints.txt`: pins `jax==0.10.2`, `jaxlib==0.10.2`.
- Create `environments/assets/panda/`: Menagerie's MJX Panda (`mjx_panda.xml`,
  `assets/`, `LICENSE`, `README.md` naming the source commit).
- Create `environments/assets/stack.xml`: floor, light, blocks, workspace
  camera; includes `panda/mjx_panda.xml`.
- Create `environments/stack_scene.py`: constants, `build_model()`,
  `home_qpos()`, block address helpers.
- Create `environments/stack_sim.py`: `Sim` (batched physics and rendering).
- Create `environments/stack_task.py`: start sampling, tower prefix, held
  block, progress (JAX).
- Create `environments/stack_control.py`: `Controller` (JAX IK).
- Create `environments/stack.py`: `BatchStack`, `Stack`.
- Create `environments/batch_driver.py`: `BatchDriver`, `make_driver`.
- Modify `environments/__init__.py`: register `stack`.
- Modify `environments/play.py`: continuous keys, several camera panels.
- Modify `train.py`: record whether the task is batched; swap the driver.
- Modify `environments/README.md`, `AGENTS.md`: document `stack`.
- Create `tests/conftest.py`, `tests/test_warp.py`,
  `tests/test_stack_scene.py`, `tests/test_stack_sim.py`,
  `tests/test_stack_task.py`, `tests/test_stack_control.py`,
  `tests/test_stack_env.py`, `tests/test_batch_driver.py`.

---

### Task 1: Install the Warp backend and check it renders on this GPU

This task is a gate. If the Panda does not render here, stop and report:
the rest of the plan depends on it.

**Files:**
- Create: `constraints.txt`, `environments/assets/panda/`,
  `tests/conftest.py`, `tests/test_warp.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mujoco`, `mujoco.mjx` with `impl='warp'`, and
  `mujoco.mjx.warp.io.create_render_context` importable in the `dreamerv3`
  env; `environments/assets/panda/mjx_panda.xml`; pytest runnable as
  `$PY -m pytest tests`.

- [ ] **Step 1: Pin JAX and install**

```bash
cd /home/max/PlanningWithWorldModels
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
cat > constraints.txt <<'EOF'
# Versions pip must not move when installing into the dreamerv3 env.
jax==0.10.2
jaxlib==0.10.2
EOF
PY=~/miniconda3/envs/dreamerv3/bin/python
$PY -m pip install -c constraints.txt mujoco==3.15.0 'mujoco-mjx[warp]==3.15.0' pytest
$PY -c "import jax, mujoco, warp; print(jax.__version__, mujoco.__version__, warp.__version__)"
$PY planning/eval_model.py --help > /dev/null && echo dreamer-ok
```

Expected: `0.10.2 3.15.0 1.17.0` and `dreamer-ok`.

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

- [ ] **Step 3: Write the test harness and the render check**

`tests/conftest.py`:

```python
"""Puts the repository and the DreamerV3 checkout on sys.path. JAX does not
preallocate GPU memory, so MuJoCo Warp can allocate beside it."""

import os
import pathlib
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MJX_WARP_ENABLED', 'true')
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
```

`tests/test_warp.py`:

```python
"""The MJX Warp backend steps and renders the Panda on this GPU."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import bvh, render, render_util
from mujoco.mjx.warp import io as warp_io

from conftest import REPO

PANDA = REPO / 'environments' / 'assets' / 'panda' / 'mjx_panda.xml'


def looking_at_the_arm():
  """Quaternion for a camera at x = 1.5 looking back at the arm."""
  x, y = np.array([0, 1, 0]), np.array([-0.5, 0, 0.866])
  mat = np.stack([x, y, np.cross(x, y)], 1)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, mat.flatten())
  return quat


def test_panda_steps_and_renders_on_gpu():
  assert jax.default_backend() == 'gpu'
  spec = mujoco.MjSpec.from_file(str(PANDA))
  spec.worldbody.add_camera(name='view', pos=[1.5, 0, 0.8],
                            quat=looking_at_the_arm())
  m = spec.compile()
  mx = mjx.put_model(m, impl='warp')
  dx = jax.vmap(lambda _: mjx.make_data(m, impl='warp'))(jnp.arange(2))
  dx = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))(mx, dx)
  ctx = warp_io.create_render_context(
      mjm=m, nworld=2, cam_res=(64, 64), use_textures=True,
      use_shadows=False, render_rgb=True, render_depth=True,
      enabled_geom_groups=[0, 1, 2])

  @jax.jit
  def draw(dx):
    dx = jax.vmap(bvh.refit_bvh, in_axes=(None, 0, None))(mx, dx, ctx.pytree())
    rgb, _, dx = jax.vmap(render.render, in_axes=(None, 0, None))(
        mx, dx, ctx.pytree())
    return jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
        ctx.pytree(), 0, rgb)

  image = np.asarray(jax.device_get(draw(dx)))
  assert image.shape == (2, 64, 64, 3)
  assert np.isfinite(image).all() and image.std() > 0.02
```

- [ ] **Step 4: Run it**

Run: `$PY -m pytest tests/test_warp.py -q -s`
Expected: `1 passed`. Also save one frame and look at it: run the same code
in a Python shell and
`PIL.Image.fromarray((image[0] * 255).astype('uint8')).resize((256, 256)).save('/tmp/panda.png')`;
the arm must be visible.

If it fails: on a Warp/CUDA error naming the architecture, check
`nvidia-smi` (driver 580 supports Blackwell) and `warp.get_cuda_devices()`;
on a JAX FFI error, record the message and try `warp-lang==1.18.0` with
`-c constraints.txt`. If it still fails, **stop and report** before any
other task.

- [ ] **Step 5: Commit**

```bash
git add constraints.txt environments/assets/panda tests/conftest.py tests/test_warp.py
git commit -m "Add the Menagerie Panda and check MJX Warp renders it on the GPU"
```

---

### Task 2: The stacking scene

**Files:**
- Create: `environments/assets/stack.xml`, `environments/stack_scene.py`
- Test: `tests/test_stack_scene.py`

**Interfaces:**
- Consumes: `environments/assets/panda/` (Task 1).
- Produces, in `environments/stack_scene.py`:
  - constants `ASSETS`, `COLORS = ('red', 'green', 'blue', 'yellow')`,
    `HALF = 0.02`, `SUBSTEPS = 10`, `CONTROL_DT = 0.05`, `HOME_ARM` (7),
    `OPEN = 0.04`, `CLOSED = 0.0`, `OPEN_WIDTH = 0.07`, `PARK` (4, 3)
  - `build_model() -> mujoco.MjModel` (cameras `workspace`, `wrist`; site
    `gripper`; bodies and free joints `block_<colour>`)
  - `home_qpos(model) -> np.ndarray (nq,)` and `home_ctrl() -> (8,)`
  - `block_qpos_adr(model)`, `block_dof_adr(model)`,
    `block_body_ids(model)`: lists of 4 ints

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_scene.py` (CPU MuJoCo as the reference for the model):

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: FAIL with `ImportError: cannot import name 'stack_scene'`.

- [ ] **Step 3: Write the scene file**

`environments/assets/stack.xml`. Blocks use `contype`/`conaffinity` 3, so
they collide with each other, the floor, the hand capsule and the finger
pads (Menagerie's arm links collide with nothing).

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

`environments/stack_scene.py`:

```python
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
PARK = np.array([[1.5, -0.3 + 0.2 * c, HALF] for c in range(len(COLORS))])


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
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_stack_scene.py -q`
Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add environments/assets/stack.xml environments/stack_scene.py tests/test_stack_scene.py
git commit -m "Add the block-stacking scene: Panda, floor, four blocks, cameras"
```

---

### Task 3: Batched simulation on the GPU

**Files:**
- Create: `environments/stack_sim.py`
- Test: `tests/test_stack_sim.py`

**Interfaces:**
- Consumes: everything in `stack_scene` (Task 2).
- Produces, in `environments/stack_sim.py`, `class Sim(num_envs, size=(64, 64))`
  with attributes `num_envs`, `size`, `mjm` (MjModel), `model` (MJX Warp
  model), `site` (int), `bodies` (int array (4,)), and methods batched over
  worlds on axis 0 (all usable inside `jax.jit`):
  - `make_data() -> Data` (every world at home, at rest)
  - `home(d) -> Data` (every world's arm and blocks back at home)
  - `set_blocks(d, positions (B, 4, 3), yaws (B, 4)) -> Data`
  - `physics(d, ctrl (B, 8), steps) -> Data` (`steps` a Python int)
  - `forward(d) -> Data`
  - `render(d) -> (workspace uint8 (B, H, W, 3), wrist uint8 (B, H, W, 3), Data)`
  - `block_pos(d) -> (B, 4, 3)`, `block_quat(d) -> (B, 4, 4)`,
    `block_vel(d) -> (B, 4, 3)`, `site_pos(d) -> (B, 3)`,
    `opening(d) -> (B,)`

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_sim.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_sim.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack_sim'`.

- [ ] **Step 3: Write the simulator**

`environments/stack_sim.py`:

```python
"""Batched block-stacking simulation on the GPU.

MJX with the Warp backend steps every world's physics (MuJoCo Warp) and
renders both cameras (MuJoCo Warp's batch renderer) as JAX functions over a
batch of worlds on axis 0. Every stage of the experiment simulates and
renders through this module.
"""

import os

os.environ.setdefault('MJX_WARP_ENABLED', 'true')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402
from mujoco.mjx._src import bvh, render, render_util  # noqa: E402
from mujoco.mjx.warp import io as warp_io  # noqa: E402

from environments import stack_scene as scene  # noqa: E402


class Sim:

  def __init__(self, num_envs, size=(64, 64)):
    self.num_envs, self.size = num_envs, tuple(size)
    self.mjm = m = scene.build_model()
    self.model = mjx.put_model(m, impl='warp')
    self.site = m.site('gripper').id
    self.bodies = np.array(scene.block_body_ids(m))
    self.qpos_adr = scene.block_qpos_adr(m)
    self.dof_adr = scene.block_dof_adr(m)
    self.cameras = (m.camera('workspace').id, m.camera('wrist').id)
    self.home_qpos = jax.device_put(scene.home_qpos(m).astype(np.float32))
    self.home_ctrl = jax.device_put(scene.home_ctrl().astype(np.float32))
    self.context = warp_io.create_render_context(
        mjm=m, nworld=num_envs, cam_res=(self.size[1], self.size[0]),
        use_textures=True, use_shadows=False, render_rgb=True,
        render_depth=True, enabled_geom_groups=[0, 1, 2])

  def make_data(self):
    """Every world at home, at rest."""
    m = self.mjm
    d = jax.vmap(lambda _: mjx.make_data(m, impl='warp'))(
        jnp.arange(self.num_envs))
    return self.forward(self.home(d))

  def home(self, d):
    b = self.num_envs
    return d.replace(
        qpos=jnp.broadcast_to(self.home_qpos, (b,) + self.home_qpos.shape),
        qvel=jnp.zeros_like(d.qvel),
        ctrl=jnp.broadcast_to(self.home_ctrl, (b,) + self.home_ctrl.shape))

  def set_blocks(self, d, positions, yaws):
    """Every world's blocks at `positions`, turned by `yaws` about z, at
    rest."""
    quat = jnp.stack([jnp.cos(yaws / 2), jnp.zeros_like(yaws),
                      jnp.zeros_like(yaws), jnp.sin(yaws / 2)], -1)
    qpos, qvel = d.qpos, d.qvel
    for c, (a, v) in enumerate(zip(self.qpos_adr, self.dof_adr)):
      qpos = qpos.at[:, a:a + 3].set(positions[:, c])
      qpos = qpos.at[:, a + 3:a + 7].set(quat[:, c])
      qvel = qvel.at[:, v:v + 6].set(0)
    return self.forward(d.replace(qpos=qpos, qvel=qvel))

  def physics(self, d, ctrl, steps):
    step = jax.vmap(mjx.step, in_axes=(None, 0))
    d = d.replace(ctrl=ctrl)
    return jax.lax.fori_loop(0, steps, lambda _, d: step(self.model, d), d)

  def forward(self, d):
    return jax.vmap(mjx.forward, in_axes=(None, 0))(self.model, d)

  def render(self, d):
    """Both cameras for every world, as uint8 RGB."""
    ctx = self.context.pytree()
    d = jax.vmap(bvh.refit_bvh, in_axes=(None, 0, None))(self.model, d, ctx)
    rgb, _, d = jax.vmap(render.render, in_axes=(None, 0, None))(
        self.model, d, ctx)
    images = []
    for camera in self.cameras:
      image = jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
          ctx, camera, rgb)
      images.append(jnp.clip(image * 255, 0, 255).astype(jnp.uint8))
    return images[0], images[1], d

  def block_pos(self, d):
    return d.xpos[:, self.bodies]

  def block_quat(self, d):
    return d.xquat[:, self.bodies]

  def block_vel(self, d):
    return jnp.stack([d.qvel[:, v:v + 3] for v in self.dof_adr], 1)

  def site_pos(self, d):
    return d.site_xpos[:, self.site]

  def opening(self, d):
    return d.qpos[:, 7] + d.qpos[:, 8]
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_sim.py -q -s`
Expected: `5 passed`, and a printed rate; write it into the commit message.
If `test_matches_cpu_mujoco` is off by more than 5 mm, print both positions:
a few mm is solver difference and the tolerance may go to 1 cm; a block in
a different place means a collision setting MuJoCo Warp does not support,
so read the warnings from `put_model` and fix `stack.xml`.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_sim.py tests/test_stack_sim.py
git commit -m "Simulate and render batches of stacking worlds on the GPU"
```

---

### Task 4: Task logic in JAX

**Files:**
- Create: `environments/stack_task.py`
- Test: `tests/test_stack_task.py`

**Interfaces:**
- Consumes: `COLORS`, `HALF`, `PARK` from `stack_scene`.
- Produces, in `environments/stack_task.py` (single-world functions; the
  environment vmaps them):
  - constants `NONE = 4`, `LIMITS = (0, 0, 400, 600, 800)` (indexed by N),
    `HOLD_STEPS = 20`, `XY_TOL = 0.015`, `Z_TOL = 0.006`, `SLOTS` (16, 2),
    `JITTER = 0.015`
  - `sample_start(key, fixed_n, prestack) -> (n, goal (4,) int32, positions (4, 3), yaws (4,))`
    with `fixed_n` 0 to sample N from {2, 3, 4}
  - `held_mask(block_pos (4, 3), site_pos (3,), opening) -> (4,) bool`
  - `tower_prefix(goal (4,), n, block_pos (4, 3), held (4,)) -> int32`
  - `progress(k_old, count, k, n, gripper_open) -> (reward, success, count)`

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_task.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import HALF, PARK
from environments.stack_task import (
    HOLD_STEPS, JITTER, NONE, SLOTS, held_mask, progress, sample_start,
    tower_prefix)

NOBODY = jnp.zeros(4, bool)


def tower(goal, xy=(0.5, 0.0)):
  """Block positions for a perfect tower in goal order, others parked."""
  pos = PARK.copy()
  for level, c in enumerate(goal):
    pos[c] = (xy[0], xy[1], HALF + 2 * HALF * level)
  return jnp.asarray(pos)


def padded(goal):
  return jnp.array(list(goal) + [NONE] * (4 - len(goal)))


def test_prefix_of_a_full_tower():
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 0, 1)), NOBODY)) == 3


def test_prefix_stops_at_the_wrong_order():
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 1, 0)), NOBODY)) == 1


def test_prefix_is_zero_when_bottom_block_is_lifted():
  pos = tower((2, 0)).at[2, 2].add(0.05).at[0, 2].add(0.05)
  assert int(tower_prefix(padded((2, 0)), 2, pos, NOBODY)) == 0


def test_prefix_ignores_held_blocks():
  held = NOBODY.at[1].set(True)
  assert int(tower_prefix(padded((2, 0, 1)), 3, tower((2, 0, 1)), held)) == 2


def test_prefix_needs_alignment():
  pos = tower((2, 0)).at[0, 0].add(0.02)
  assert int(tower_prefix(padded((2, 0)), 2, pos, NOBODY)) == 1


def test_held_needs_a_block_between_closed_fingers():
  pos = jnp.asarray(PARK).at[3].set(jnp.array([0.5, 0.0, 0.1]))
  site = jnp.array([0.5, 0.0, 0.1])
  assert held_mask(pos, site, 0.04).tolist() == [False, False, False, True]
  assert not held_mask(pos, site, 0.08).any()   # Open.
  assert not held_mask(pos, site, 0.0).any()    # Closed on nothing.


def test_progress_rewards_changes_in_prefix():
  r, s, count = progress(1, 0, 2, 3, False)
  assert (float(r), bool(s)) == (1.0, False)
  r, s, count = progress(2, count, 1, 3, False)
  assert float(r) == -1.0


def test_success_needs_the_hold():
  k_old, count = 1, 0
  rewards, successes = [], []
  for _ in range(HOLD_STEPS + 1):
    r, s, count = progress(k_old, count, 2, 2, True)
    k_old = 2
    rewards.append(float(r))
    successes.append(bool(s))
  assert rewards[0] == 1.0 and not any(successes[:-1])
  assert successes[-1] and rewards[-1] == 1.0


def test_hold_restarts_when_gripper_closes():
  count = 0
  for _ in range(HOLD_STEPS - 1):
    _, _, count = progress(2, count, 2, 2, True)
  _, _, count = progress(2, count, 2, 2, False)
  assert int(count) == 0


def test_starts_are_separated_and_in_the_workspace():
  keys = jax.random.split(jax.random.PRNGKey(0), 256)
  n, goal, pos, _ = jax.vmap(sample_start, (0, None, None))(keys, 0, 0.0)
  n, goal, pos = map(np.asarray, (n, goal, pos))
  assert set(n.tolist()) == {2, 3, 4}
  lo, hi = SLOTS.min(0) - JITTER, SLOTS.max(0) + JITTER
  for i in range(len(n)):
    blocks = goal[i][:n[i]]
    assert (goal[i][n[i]:] == NONE).all()
    xy = pos[i][blocks][:, :2]
    assert ((xy >= lo) & (xy <= hi)).all()
    gaps = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    assert gaps[np.triu_indices(len(blocks), 1)].min() >= 0.07
    absent = [c for c in range(4) if c not in blocks]
    np.testing.assert_allclose(pos[i][absent], PARK[absent])


def test_prestacked_starts_stack_two_or_three():
  keys = jax.random.split(jax.random.PRNGKey(1), 256)
  n, goal, pos, _ = jax.vmap(sample_start, (0, None, None))(keys, 0, 1.0)
  n, goal, pos = map(np.asarray, (n, goal, pos))
  for i in range(len(n)):
    zs = pos[i][goal[i][:n[i]]][:, 2]
    assert int((zs > HALF + 1e-4).sum()) + 1 in (2, 3)


def test_fixed_n():
  keys = jax.random.split(jax.random.PRNGKey(2), 32)
  n, *_ = jax.vmap(sample_start, (0, None, None))(keys, 3, 0.3)
  assert set(np.asarray(n).tolist()) == {3}
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_task.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack_task'`.

- [ ] **Step 3: Write the task logic**

`environments/stack_task.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_task.py -q`
Expected: `12 passed`. The trickiest code is the rank and slot assignment in
`sample_start`; if `test_starts_are_separated_and_in_the_workspace` fails,
print `rank`, `slot` and `xy` for the failing key before changing anything.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_task.py tests/test_stack_task.py
git commit -m "Add block-stacking task logic in JAX: starts, prefix, reward"
```

---

### Task 5: End-effector controller in JAX

**Files:**
- Create: `environments/stack_control.py`
- Test: `tests/test_stack_control.py`

**Interfaces:**
- Consumes: `Sim` (Task 3), `stack_scene` constants.
- Produces, in `environments/stack_control.py`:
  - constants `LO = np.array([0.30, -0.22, 0.015])`,
    `HI = np.array([0.70, 0.22, 0.35])`, `MAX_STEP = 0.02`,
    `MAX_YAW = 0.2`, `YAW_LIMIT = np.pi / 2`, `HOLD = (0, 0, 0, 0, -1)`
  - `class Controller(sim)` with batched methods usable inside `jax.jit`:
    `ee(d) -> (pos (B, 3), yaw (B,))`,
    `start(d) -> (target (B, 3), yaw (B,))` (current pose clipped into the
    bounds), and
    `act(d, target, yaw, action (B, 5)) -> (ctrl (B, 8), target, yaw)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_control.py`:

```python
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


def test_moves_along_x():
  arm = Arm()
  arm.settle()
  start = arm.pos()
  for _ in range(5):
    arm.apply([1.0, 0, 0, 0, -1])
  arm.settle(10)
  moved = arm.pos() - start
  assert moved[0] == pytest.approx(0.10, abs=0.01)
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_control.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack_control'`.

- [ ] **Step 3: Write the controller**

`environments/stack_control.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_control.py -q`
Expected: `5 passed`. If the grasp fails, print the site's z and the
block's z after the descent: the finger pads sit about 1 cm from the site,
so lowering `LO[2]` or the descent target by up to 1 cm is the expected
fix, not changing friction.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_control.py tests/test_stack_control.py
git commit -m "Add end-effector control by damped least-squares IK in JAX"
```

---

### Task 6: `BatchStack` and the `stack` environment

**Files:**
- Create: `environments/stack.py`
- Modify: `environments/__init__.py`
- Test: `tests/test_stack_env.py`

**Interfaces:**
- Consumes: `Sim` (Task 3), task functions (Task 4), `Controller` and
  `HOLD` (Task 5).
- Produces, in `environments/stack.py`:
  - `obs_space(size)` and `act_space()`: the spaces, without building a
    simulator.
  - `class BatchStack(num_envs, size=(64, 64), n=0, prestack=0.3, seed=0)`
    with `step(acts) -> obs`, where `acts` is `{'reset': bool (B,),
    'action': float32 (B, 5)}` and `obs` a dict of NumPy arrays with a
    leading batch axis: `workspace`, `wrist` (uint8 (B, 64, 64, 3)), `arm`
    (float32 (B, 19)), `goal` (int32 (B, 4)), `reward`, `is_first`,
    `is_last`, `is_terminal`, `log/prefix`, `log/n`, `log/success`,
    `log/sim_steps`; `privileged() -> {'blocks': float32 (B, 4, 11),
    'held': int32 (B,)}`; attributes `sim`, `control`, `state` (dict with
    `data`, `key`, `target`, `yaw`, `goal`, `n`, `k`, `count`, `steps`,
    `sim_steps`). A world restarts when its `reset` flag is set or its
    episode ended on the previous step (and on the first call); its first
    observation has `is_first` and reward 0.
  - `class Stack(embodied.Env)` registered as `stack`:
    `Stack(task, batch=1, size=(64, 64), n=0, prestack=0.3, seed=None)`,
    with `BATCHED = True`, `obs_space`, `act_space`, `step(action)` (one
    world; needs `batch=1`), `step_batch(acts)` and `privileged()`
    (batched). The simulator is built on the first step, so reading the
    spaces costs nothing.

- [ ] **Step 1: Write the failing tests**

`tests/test_stack_env.py`:

```python
import jax
import numpy as np
import pytest

import environments
from environments.stack import BatchStack, obs_space
from environments.stack_scene import HALF
from environments.stack_task import HOLD_STEPS, LIMITS

B = 4


def acts(action=(0, 0, 0, 0, -1), reset=False, batch=B):
  return {'reset': np.full(batch, reset),
          'action': np.tile(np.asarray(action, np.float32), (batch, 1))}


def check_obs(obs, batch=B):
  for key, space in obs_space((64, 64)).items():
    value = np.asarray(obs[key])
    assert value.shape == (batch,) + space.shape, key
    assert value.dtype == space.dtype, key
  assert np.isfinite(obs['arm']).all()


def teleport(env, world, placements):
  """Move blocks of one world: {colour: (x, y, z)}; others stay put."""
  pos = np.asarray(jax.device_get(env.sim.block_pos(env.state['data'])))
  for c, xyz in placements.items():
    pos[world, c] = xyz
  env.state['data'] = env.sim.set_blocks(
      env.state['data'], jax.device_put(pos.astype(np.float32)),
      jax.device_put(np.zeros((env.num_envs, 4), np.float32)))


def build_tower(env, world=0):
  """Teleport the world's episode blocks into its goal tower."""
  goal = np.asarray(jax.device_get(env.state['goal']))[world]
  n = int(jax.device_get(env.state['n'])[world])
  teleport(env, world, {int(c): (0.5, 0.15, HALF + 2 * HALF * level)
                        for level, c in enumerate(goal[:n])})


@pytest.fixture(scope='module')
def env():
  return BatchStack(B, seed=0)


def test_first_observation(env):
  obs = env.step(acts(reset=True))
  check_obs(obs)
  assert obs['is_first'].all() and not obs['is_last'].any()
  assert (obs['reward'] == 0).all()
  for i in range(B):
    n = int(obs['log/n'][i])
    assert (obs['goal'][i][:n] < 4).all() and (obs['goal'][i][n:] == 4).all()


def test_step_observation(env):
  env.step(acts(reset=True))
  obs = env.step(acts((0.5, 0, 0, 0, -1)))
  check_obs(obs)
  assert not obs['is_first'].any()


def test_reset_starts_at_rest():
  env = BatchStack(64, seed=1, prestack=0.5)
  env.step(acts(reset=True, batch=64))
  blocks = env.privileged()['blocks']
  present = blocks[..., 0] > 0
  assert np.abs(blocks[..., 8:11][present]).max() < 0.02


def test_privileged_marks_present_blocks(env):
  obs = env.step(acts(reset=True))
  blocks = env.privileged()['blocks']
  assert blocks.shape == (B, 4, 11)
  for i in range(B):
    present = set(np.flatnonzero(blocks[i, :, 0]).tolist())
    n = int(obs['log/n'][i])
    assert present == set(obs['goal'][i][:n].tolist())


def test_building_the_tower_succeeds():
  env = BatchStack(2, seed=3, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=2))
  build_tower(env, 0)
  total = 0.0
  for _ in range(HOLD_STEPS + 5):
    obs = env.step(acts(batch=2))
    total += float(obs['reward'][0])
    if obs['is_last'][0]:
      break
  assert obs['is_last'][0] and obs['is_terminal'][0]
  assert int(obs['log/success'][0]) == 1
  assert total == pytest.approx(2.0)  # Prefix 1 -> 2, plus success.
  assert not obs['is_last'][1]


def test_toppling_gives_negative_reward():
  env = BatchStack(1, seed=4, n=3, prestack=0.0)
  env.step(acts(reset=True, batch=1))
  build_tower(env, 0)
  env.step(acts(batch=1))
  top = int(np.asarray(jax.device_get(env.state['goal']))[0, 2])
  teleport(env, 0, {top: (0.6, -0.1, HALF)})
  obs = env.step(acts(batch=1))
  assert float(obs['reward'][0]) == pytest.approx(-1.0)
  assert not obs['is_last'][0]


def test_time_limit_ends_episode():
  env = BatchStack(1, seed=5, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=1))
  for _ in range(LIMITS[2]):
    obs = env.step(acts(batch=1))
  assert obs['is_last'][0] and not obs['is_terminal'][0]
  assert env.step(acts(batch=1))['is_first'][0]


def test_worlds_reset_independently():
  env = BatchStack(2, seed=6, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=2))
  for _ in range(5):
    env.step(acts(batch=2))
  before = int(env.step(acts(batch=2))['log/sim_steps'][1])
  flagged = acts(batch=2)
  flagged['reset'][0] = True
  obs = env.step(flagged)
  assert obs['is_first'][0] and not obs['is_first'][1]
  assert int(obs['log/sim_steps'][1]) == before + 1


def test_same_seed_same_start():
  a, b = BatchStack(2, seed=7), BatchStack(2, seed=7)
  oa = a.step(acts(reset=True, batch=2))
  ob = b.step(acts(reset=True, batch=2))
  np.testing.assert_array_equal(oa['goal'], ob['goal'])
  np.testing.assert_allclose(a.privileged()['blocks'],
                             b.privileged()['blocks'], atol=1e-5)


def test_runs_under_transfer_guard():
  with jax.transfer_guard('disallow'):
    env = BatchStack(2, seed=8)
    env.step(acts(reset=True, batch=2))
    env.step(acts((0.3, 0, 0, 0, -1), batch=2))
    env.privileged()


def test_single_world_env():
  env = environments.make('stack', seed=0)
  assert env.obs_space['workspace'].shape == (64, 64, 3)
  obs = env.step({'reset': True, 'action': np.zeros(5, np.float32)})
  assert obs['is_first'] and obs['workspace'].shape == (64, 64, 3)
  obs = env.step({'reset': False, 'action': np.zeros(5, np.float32)})
  assert not obs['is_first']
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.stack'`.

- [ ] **Step 3: Write the environment**

`environments/stack.py`:

```python
"""Block stacking with a Franka Panda, acting through end-effector deltas.

BatchStack steps a batch of worlds together on the GPU. Each episode samples
N blocks (2-4) and a goal order; the agent sees a workspace camera, a wrist
camera, the arm's state and the goal, and moves the gripper by up to 2 cm
and 0.2 rad of yaw per 20 Hz control step. Scalar diagnostics are log/ keys;
the full block state comes from privileged(). Stack is the embodied.Env
around it, which DreamerV3 and play.py use.
"""

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np

from environments import stack_scene as scene
from environments import stack_task as task
from environments.stack_control import HOLD, Controller
from environments.stack_sim import Sim

ARM_DIM = 19  # 7 joint angles, 7 velocities, gripper xyz, yaw, opening.
SETTLE = 60  # Physics steps (0.3 s) to let a new layout come to rest.


def obs_space(size):
  image = elements.Space(np.uint8, tuple(size) + (3,))
  return {
      'workspace': image,
      'wrist': image,
      'arm': elements.Space(np.float32, (ARM_DIM,)),
      'goal': elements.Space(np.int32, (len(scene.COLORS),), 0, task.NONE + 1),
      'reward': elements.Space(np.float32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
      'log/prefix': elements.Space(np.int32),
      'log/n': elements.Space(np.int32),
      'log/success': elements.Space(np.int32),
      'log/sim_steps': elements.Space(np.int32),
  }


def act_space():
  return {
      'reset': elements.Space(bool),
      'action': elements.Space(np.float32, (5,), -1, 1),
  }


class BatchStack:

  def __init__(self, num_envs, size=(64, 64), n=0, prestack=0.3, seed=0):
    self.num_envs = num_envs
    self.fixed_n, self.prestack = int(n), float(prestack)
    # Building moves the model and constants to the GPU once. DreamerV3
    # forbids implicit transfers, so allow them here; every step moves its
    # inputs and outputs explicitly.
    with jax.transfer_guard('allow'):
      self.sim = Sim(num_envs, size)
      self.control = Controller(self.sim)
      self.limits = jax.device_put(np.asarray(task.LIMITS, np.int32))
      self.hold = jax.device_put(np.asarray(HOLD, np.float32))
      self.state = self._init(jax.random.PRNGKey(seed))
    self._restart = jax.jit(self._restart_impl)
    self._advance = jax.jit(self._advance_impl)
    self._privileged = jax.jit(self._privileged_impl)
    self.done = np.ones(num_envs, bool)

  def step(self, acts):
    """One control step for every world; ended or flagged worlds restart."""
    restart = np.asarray(acts['reset'], bool) | self.done
    mask = jax.device_put(restart)
    if restart.any():
      self.state = self._restart(self.state, mask)
    action = jax.device_put(np.asarray(acts['action'], np.float32))
    self.state, obs = self._advance(self.state, action, mask)
    obs = {k: np.asarray(v) for k, v in jax.device_get(obs).items()}
    self.done = obs['is_last'].copy()
    return obs

  def privileged(self):
    """The simulator's block state, which the agent does not observe."""
    out = jax.device_get(self._privileged(self.state))
    return {k: np.asarray(v) for k, v in out.items()}

  def _init(self, key):
    b = self.num_envs
    d = self.sim.make_data()
    zeros = jnp.zeros(b, jnp.int32)
    target, yaw = self.control.start(d)
    return dict(
        data=d, key=key, target=target, yaw=yaw,
        goal=jnp.full((b, len(scene.COLORS)), task.NONE, jnp.int32),
        n=zeros + 2, k=zeros, count=zeros, steps=zeros, sim_steps=zeros)

  def _restart_impl(self, state, mask):
    """New episodes for the worlds in `mask`; the others are untouched."""
    key, sub = jax.random.split(state['key'])
    keys = jax.random.split(sub, self.num_envs)
    n, goal, pos, yaws = jax.vmap(task.sample_start, (0, None, None))(
        keys, self.fixed_n, self.prestack)
    fresh = self.sim.set_blocks(self.sim.home(state['data']), pos, yaws)
    fresh = self.sim.physics(fresh, fresh.ctrl, SETTLE)
    d = state['data']
    m = mask[:, None]
    d = self.sim.forward(d.replace(
        qpos=jnp.where(m, fresh.qpos, d.qpos),
        qvel=jnp.where(m, fresh.qvel, d.qvel),
        ctrl=jnp.where(m, fresh.ctrl, d.ctrl)))
    target, yaw = self.control.start(d)

    def pick(new, old):
      return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)),
                       new, old)

    return dict(
        state, data=d, key=key,
        target=pick(target, state['target']), yaw=pick(yaw, state['yaw']),
        goal=pick(goal, state['goal']), n=pick(n, state['n']))

  def _advance_impl(self, state, action, first):
    """One control step. The first step of a new episode holds still and
    reports the starting state with no reward."""
    action = jnp.where(first[:, None], self.hold, action)
    d = state['data']
    ctrl, target, yaw = self.control.act(
        d, state['target'], state['yaw'], action)
    d = self.sim.physics(d, ctrl, scene.SUBSTEPS)
    workspace, wrist, d = self.sim.render(d)
    pos, opening = self.sim.block_pos(d), self.sim.opening(d)
    held = jax.vmap(task.held_mask)(pos, self.sim.site_pos(d), opening)
    k = jax.vmap(task.tower_prefix)(state['goal'], state['n'], pos, held)
    reward, success, count = jax.vmap(task.progress)(
        state['k'], state['count'], k, state['n'],
        opening > scene.OPEN_WIDTH)
    reward = jnp.where(first, 0.0, reward)
    success = success & ~first
    count = jnp.where(first, 0, count)
    steps = jnp.where(first, 0, state['steps'] + 1)
    sim_steps = state['sim_steps'] + jnp.where(first, 0, 1)
    last = success | (steps >= self.limits[state['n']])
    ee_pos, ee_yaw = self.control.ee(d)
    arm = jnp.concatenate([
        d.qpos[:, :7], d.qvel[:, :7], ee_pos, ee_yaw[:, None],
        opening[:, None]], -1)
    obs = {
        'workspace': workspace, 'wrist': wrist,
        'arm': arm.astype(jnp.float32), 'goal': state['goal'],
        'reward': reward.astype(jnp.float32),
        'is_first': first, 'is_last': last, 'is_terminal': success,
        'log/prefix': k.astype(jnp.int32), 'log/n': state['n'],
        'log/success': success.astype(jnp.int32),
        'log/sim_steps': sim_steps.astype(jnp.int32),
    }
    state = dict(state, data=d, target=target, yaw=yaw, k=k, count=count,
                 steps=steps, sim_steps=sim_steps)
    return state, obs

  def _privileged_impl(self, state):
    d = state['data']
    colors = jnp.arange(len(scene.COLORS))
    present = (colors[None, :, None] == state['goal'][:, None, :]).any(-1)
    pos = self.sim.block_pos(d)
    blocks = jnp.concatenate([
        present[..., None].astype(jnp.float32), pos,
        self.sim.block_quat(d), self.sim.block_vel(d)], -1)
    blocks = blocks * present[..., None]
    held = jax.vmap(task.held_mask)(
        pos, self.sim.site_pos(d), self.sim.opening(d)) & present
    held = jnp.where(held.any(-1), jnp.argmax(held, -1), -1)
    return {'blocks': blocks, 'held': held.astype(jnp.int32)}


class Stack(embodied.Env):

  BATCHED = True

  def __init__(self, task_name, batch=1, size=(64, 64), n=0, prestack=0.3,
               seed=None):
    del task_name
    self.batch, self.size = int(batch), tuple(size)
    self.options = dict(n=n, prestack=prestack)
    self.seed = int(np.random.SeedSequence().entropy % 2 ** 31
                    if seed is None else seed)
    self._env = None

  @property
  def env(self):
    if self._env is None:
      self._env = BatchStack(self.batch, self.size, seed=self.seed,
                             **self.options)
    return self._env

  @property
  def obs_space(self):
    return obs_space(self.size)

  @property
  def act_space(self):
    return act_space()

  def step(self, action):
    assert self.batch == 1, 'Use step_batch() for more than one world.'
    acts = {k: np.asarray(v)[None] for k, v in action.items()}
    return {k: v[0] for k, v in self.env.step(acts).items()}

  def step_batch(self, acts):
    return self.env.step(acts)

  def privileged(self):
    return self.env.privileged()
```

In `environments/__init__.py`, add to `REGISTRY`:

```python
REGISTRY = {
    'catch': 'environments.catch:Catch',
    'stack': 'environments.stack:Stack',
    'tetris': 'environments.tetris:Tetris',
}
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_env.py -q`
Expected: `11 passed`. The bottom goal block starts on the table, so k is 1
at reset; `test_building_the_tower_succeeds` expects +1 for k going from 1
to 2 and +1 for success. If `test_runs_under_transfer_guard` fails inside a
step, find the NumPy value the jitted function captured or received and
`jax.device_put` it in `__init__`.

- [ ] **Step 5: Run the whole suite**

Run: `$PY -m pytest tests -q`
Expected: all 38 tests pass.

- [ ] **Step 6: Commit**

```bash
git add environments/stack.py environments/__init__.py tests/test_stack_env.py
git commit -m "Add the batched stack environment for raw end-effector actions"
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

- [ ] **Step 1: Let JAX and MuJoCo Warp share the GPU**

At the top of `environments/play.py`, before `import environments`, add
`import os` to the imports and:

```python
# The stack environment simulates on the GPU in this process.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MJX_WARP_ENABLED', 'true')
```

- [ ] **Step 2: Draw every camera in the observation**

Replace the start of `render` so it draws every uint8 HxWx3 observation, in
sorted key order, and keep the rest (emulator screen, canvas) as it is:

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

- [ ] **Step 3: Add the stack keys and continuous actions**

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

In `Player`, replace `reset` and `press`, and add `zero_action`:

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

- [ ] **Step 4: Check the existing environments still play**

Run `$PY environments/play.py catch` and press Left/Right, then Escape. Run
`$PY environments/play.py tetris` and press a digit. Both behave as before.

- [ ] **Step 5: Play `stack` by hand**

Run: `$PY environments/play.py stack --scale 6`
Check, and say in the commit message what was seen: both camera panels
show; W/S/A/D/R/F move the gripper the right way in the workspace camera;
Space closes the gripper; you can pick up one block and set it on another;
the status line's reward shows +1 when the tower grows.

- [ ] **Step 6: Commit**

```bash
git add environments/play.py
git commit -m "Play the stack environment by hand with both cameras"
```

---

### Task 8: DreamerV3 steps the batch through a batched driver

**Files:**
- Create: `environments/batch_driver.py`
- Modify: `train.py`
- Test: `tests/test_batch_driver.py`

**Interfaces:**
- Consumes: `Stack.BATCHED`, `Stack.step_batch` (Task 6); DreamerV3's
  `embodied.Driver` interface (`reset(init_policy=None)`, `close()`,
  `on_step(callback)`, `__call__(policy, steps=0, episodes=0)`, attribute
  `act_space`); `train.make_env`.
- Produces, in `environments/batch_driver.py`: `class BatchDriver` with
  `embodied.Driver`'s interface, built from the same `make_env_fns` (it
  calls the first with `batch=len(make_env_fns)`), and
  `make_driver(original, batched)` returning a factory that builds
  `BatchDriver` when `batched()` is true and `original` otherwise. In
  `train.py`: the module global `BATCHED` and `embodied.Driver` replaced by
  that factory.

- [ ] **Step 1: Write the failing tests**

`tests/test_batch_driver.py` uses a small batched fake, so it runs on the
CPU, and checks the driver against DreamerV3's own on the same fake:

```python
import elements
import embodied
import numpy as np

from environments.batch_driver import BatchDriver


class Counter(embodied.Env):
  """Each world counts up from 0 and ends after `length` steps."""

  BATCHED = True

  def __init__(self, batch=1, length=3):
    self.batch, self.length = batch, length
    self.t = np.zeros(batch, int)
    self.done = np.ones(batch, bool)

  @property
  def obs_space(self):
    return {'x': elements.Space(np.float32),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool), 'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool)}

  @property
  def act_space(self):
    return {'reset': elements.Space(bool),
            'action': elements.Space(np.float32, (2,), -1, 1)}

  def step_batch(self, acts):
    restart = acts['reset'] | self.done
    self.t = np.where(restart, 0, self.t + 1)
    self.done = self.t >= self.length
    return {'x': self.t.astype(np.float32),
            'reward': np.ones(self.batch, np.float32),
            'is_first': restart.copy(), 'is_last': self.done.copy(),
            'is_terminal': self.done.copy()}

  def step(self, action):
    acts = {k: np.asarray(v)[None] for k, v in action.items()}
    return {k: v[0] for k, v in self.step_batch(acts).items()}


def policy(carry, obs):
  return carry, {'action': np.zeros((len(obs['x']), 2), np.float32)}, {}


def record(driver, steps):
  seen = []
  driver.on_step(lambda tran, i: seen.append(
      (i, float(tran['x']), bool(tran['is_first']))))
  driver.reset()
  driver(policy, steps=steps)
  return sorted(seen)


def test_matches_dreamers_driver():
  ours = record(BatchDriver([lambda **kw: Counter(**kw)] * 3), 12)
  theirs = record(embodied.Driver([lambda: Counter()] * 3, parallel=False), 12)
  assert ours == theirs


def test_builds_one_batched_env():
  calls = []

  def make(**kw):
    calls.append(kw)
    return Counter(**kw)

  BatchDriver([make] * 5)
  assert calls == [{'batch': 5}]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_batch_driver.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'environments.batch_driver'`.

- [ ] **Step 3: Write the driver**

`environments/batch_driver.py`:

```python
"""A DreamerV3 driver that steps one batched environment.

embodied.Driver steps environments one at a time. Environments with
BATCHED = True step all their worlds at once on the GPU, so BatchDriver
makes one of them with batch=<number of environments> and calls its
step_batch(). It keeps embodied.Driver's interface and transitions, so the
rest of DreamerV3's training loop is unchanged.
"""

import elements
import numpy as np


class BatchDriver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    del parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    env = make_env_fns[0](batch=self.length)
    # DreamerV3's wrappers keep the environment they wrap as .env; their
    # action normalisation and clipping are identities for [-1, 1] actions.
    while not hasattr(env, 'step_batch'):
      env = env.env
    self.env = env
    self.act_space = env.act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    self.env.close()

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    obs = self.env.step_batch(self.acts)
    logs = {k: v for k, v in obs.items() if k.startswith('log/')}
    obs = {k: v for k, v in obs.items() if not k.startswith('log/')}
    self.carry, acts, outs = policy(self.carry, obs, **self.kwargs)
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    self.acts = {**acts, 'reset': obs['is_last'].copy()}
    trans = {**obs, **acts, **outs, **logs}
    for i in range(self.length):
      trn = elements.tree.map(lambda x: x[i], trans)
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)


def make_driver(original, batched):
  """A Driver factory: BatchDriver when `batched()` says the environment
  is batched, `original` otherwise."""

  def driver(make_env_fns, parallel=True, **kwargs):
    if batched():
      return BatchDriver(make_env_fns, parallel, **kwargs)
    return original(make_env_fns, parallel, **kwargs)

  return driver
```

- [ ] **Step 4: Run the driver tests**

Run: `$PY -m pytest tests/test_batch_driver.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Hook it into `train.py`**

DreamerV3's `embodied.run.train` calls `make_agent()` first, which builds
one environment for the agent's spaces through `make_env`, and creates the
driver after. So `make_env` can record whether the task is batched before
the driver is made. Check this order in
`third_party/dreamerv3/embodied/run/train.py` (the agent is built before
`embodied.Driver(fns, ...)`) before relying on it.

In `train.py`, replace `make_env` and the lines after it with:

```python
BATCHED = False


def make_env(config, index, **overrides):
  global BATCHED
  suite, task = config.task.split('_', 1)
  if suite != 'custom':
    return builtin_make_env(config, index, **overrides)
  import environments
  ctor = environments.lookup(task)
  BATCHED = getattr(ctor, 'BATCHED', False)
  options = config.env.get(getattr(ctor, 'OPTIONS', 'custom'), {})
  kwargs = {**options, **overrides}
  if kwargs.pop('use_seed', False):
    kwargs['seed'] = hash((config.seed, index)) % (2 ** 32 - 1)
  return main.wrap_env(ctor(task, **kwargs), config)


# main.main() looks make_env up by name when it builds the environments, and
# DreamerV3's run scripts look embodied.Driver up when they make the driver.
main.make_env = make_env
embodied.Driver = batch_driver.make_driver(embodied.Driver, lambda: BATCHED)
```

and add, after `from dreamerv3 import main  # noqa: E402`:

```python
import embodied  # noqa: E402
from environments import batch_driver  # noqa: E402
```

Run: `$PY train.py --help 2>&1 | head -3` and
`$PY planning/eval_model.py --help > /dev/null && echo ok`.
Expected: no tracebacks; `ok`.

- [ ] **Step 6: Short DreamerV3 run on the GPU**

```bash
cd /home/max/PlanningWithWorldModels
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
scratch=/tmp/claude-1000/-home-max-PlanningWithWorldModels/stack_debug
rm -rf $scratch
$PY train.py --task custom_stack --configs debug --logdir $scratch \
  --jax.platform cuda --jax.prealloc False --run.steps 3500 2>&1 | tail -20
grep -c . $scratch/metrics.jsonl
grep -o '"epstats/log/[a-z_]*/avg"' $scratch/metrics.jsonl | sort -u
ls $scratch/scope | head
```

`run.steps` counts steps over all 4 worlds, and an episode lasts at most
800 control steps (N = 4), so 3500 steps ends at least one episode in every
world. `--configs debug` uses a tiny model; `--jax.platform cuda` keeps it
on the GPU next to the simulator.

Expected: no traceback; `episode/score` lines in `metrics.jsonl`;
`epstats/log/n/avg`, `epstats/log/prefix/avg`, `epstats/log/sim_steps/avg`
and `epstats/log/success/avg` among the keys; policy videos for both cameras
(`epstats-policy_workspace.mp4`, `epstats-policy_wrist.mp4`) in `scope/`.
Write the training fps from the log's last lines into the commit message.

- [ ] **Step 7: Commit**

```bash
git add environments/batch_driver.py train.py tests/test_batch_driver.py
git commit -m "Let DreamerV3 step batched GPU environments through train.py"
```

---

### Task 9: Document the environment

**Files:**
- Modify: `environments/README.md`, `AGENTS.md`

- [ ] **Step 1: Document `stack`**

Add under "## Environments" in `environments/README.md`:

```markdown
- `stack`: a Franka Panda stacks 2-4 coloured blocks in a goal order. The
  agent sees a workspace camera and a wrist camera (64x64 RGB), the arm's
  state and the goal, and moves the gripper by end-effector deltas at 20 Hz.
  Physics and rendering run on the GPU (MJX with the Warp backend) for a
  batch of worlds at once: `train.py` steps them through
  `batch_driver.BatchDriver`, so the agent and the environment share the
  GPU; pass `--jax.prealloc False`. Options: `n` fixes N (0 samples 2-4),
  `prestack` is the share of starts with a stack already built, `size` the
  camera resolution. Block poses are not observed; `privileged()` returns
  them. Install with `pip install -c constraints.txt mujoco==3.15.0
  'mujoco-mjx[warp]==3.15.0'`.
```

and under "## Playing by hand":

```markdown
    python environments/play.py stack --scale 6
```

- [ ] **Step 2: Note the GPU environment in AGENTS.md**

Under "## Python environments" in `AGENTS.md`, add:

```markdown
- `custom_stack` simulates and renders on the GPU inside the training
  process (MJX Warp backend). Train it with `--jax.prealloc False`, and run
  `python -m pytest tests` only while no other GPU process is up.
```

- [ ] **Step 3: Commit**

```bash
git add environments/README.md AGENTS.md
git commit -m "Document the stack environment and its GPU use"
```

---

## Later stages

Each gets its own plan when the stage before it is done, using what it
showed.

- **Stage 2: Skills (GPU).** `skills/env.py` (pick and place tasks on
  `Sim`, sharing `sample_start` and the controller), `skills/teacher.py`
  (Brax PPO state teachers with a terminate output, thousands of worlds),
  `skills/student.py` (CNN image students by DAgger on `BatchStack`'s
  renderer), `skills/policy.py` (loading a student as a JAX function).
  Gates: teacher and student each at least 90% of 500 episodes. Installs
  `brax` with `-c constraints.txt`.
- **Stage 3: Raw-action agent (GPU).** DreamerV3 on `custom_stack` through
  `BatchDriver`: an N = 2 pilot for learning speed, then the main run over
  N in {2, 3, 4}.
- **Stage 4: Option agent (GPU).** `environments/stack_options.py`
  (`custom_stack_options`, batched: the 9 options run as student skills
  inside the jitted step; a world whose option has ended keeps its state
  while the rest of the batch finishes theirs; validity masks in
  `privileged()`, 10-step cost for invalid options, `log/duration`, summed
  reward, time-limit cut-off); the N = 2 pilot sets the decision budget from
  the mean option duration; the main run at stage 3's control-step budget.
- **Stage 5: Planning and evaluation.** iCEM and sampled MuZero in
  `planning/search.py` for both action spaces; `eval_model.py` for batched
  custom environments, continuous actions and success by N;
  `planning/model_error.py` with pose probes; the results table.
