# Block Stacking Skills Implementation Plan (Stage 2 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Trained `pick(colour)` and `place_on(colour | table)` skills that act
from the two cameras, each succeeding in at least 90% of 500 test episodes,
ready for the option environment of stage 4.

**Architecture:** A batched Brax environment per skill (`skills/env.py`)
runs on stage 1's GPU simulator (`Sim`, `Controller`, `sample_start`) and
keeps its own per-world resets. A state teacher is trained on it with Brax
PPO from privileged state (`skills/teacher.py`). An image student, a CNN over
both cameras, is distilled from the teacher with DAgger on the same simulator
and renderer (`skills/student.py`). `skills/evaluate.py` runs the 90% gates.
`skills/policy.py` loads trained skills as JAX functions. A scripted
controller (`skills/scripted.py`) checks that each task can be solved and
that success is detected, before any learning.

**Tech Stack:** JAX 0.10.2, MJX Warp (stage 1), Brax 0.14.2 PPO, Flax 0.12.8
linen, Optax, NumPy, pytest. Local RTX 5070 Ti.

**Spec:** `docs/superpowers/specs/2026-10-07-block-stacking-options-design.md`
(section "Skills"; stage 2 of "Build order").

## Where to run

Measured on the local RTX 5070 Ti with stage 1's simulator:
- **Without cameras (teachers):** physics plus IK runs at 32,600 control steps/s
  with 8,192 worlds and 19,300 with 4,096.
- **With both cameras (students):** about 7,000 steps/s at 256 worlds.

Stage 2's estimated compute locally:
- **Teachers:** two, at about 100M PPO steps each, roughly 1–1.5 h apiece
  including network updates.
- **Students:** DAgger, about 0.5 h each.
- **Total:** about 4 h.

Delta's A100s are probably no faster per GPU for this workload. DeltaAI's
GH200s might give 2–3× (an estimate, not measured). Either one would need a
new software environment (JAX 0.10.2, MJX Warp, Brax; DeltaAI is aarch64),
sync and queue time, and a Duo login per sync, all to save at most about 2 h.

**Run stage 2 locally.** Delta becomes worth it in stages 3–4, where several
DreamerV3 seeds can run in parallel. Task 8 records measured times, so that
decision is made on numbers.

## Global Constraints

- Code style from AGENTS.md: 2-space indentation, single quotes, about 80
  columns, no type hints, short docstrings, comments describe the code as
  it is.
- Do not edit `third_party/`.
- `PY=~/miniconda3/envs/dreamerv3/bin/python`. Install with
  `-c constraints.txt`, so JAX stays at 0.10.2.
- One GPU process at a time: check that `nvidia-smi` lists no compute
  processes before GPU tests or training.
- Physics and rendering go through `environments/stack_sim.py`; skills and
  the option environment see the same renderer.
- **Skills (spec):**
  - **Pick(c):** block c grasped and lifted 5 cm clear of anything below it.
  - **Place(target):** the target is a colour or the table; the held block is
    released and resting on the target, with the gripper open and raised 5 cm.
- **Actions:** each skill outputs the 5 raw action dimensions plus a
  terminate output; `action[5] > 0` ends the skill. Ending earns
  `SUCCESS_BONUS = 5` if the success condition holds, and `-FAIL_PENALTY = -1`
  if not.
- **Timeout:** a skill runs for at most 150 control steps (`EPISODE = 150`).
- **Teacher:** PPO from privileged state, with its own dense shaped reward
  that the stacking agents never see.
- **Student:** a CNN on `workspace`, `wrist`, `arm` and the target (plus the
  fraction of the episode elapsed), trained by DAgger with the teacher
  labelling each visited state.
- **Starts:** drawn from the stacking start distribution (`sample_start`,
  `prestack=0.3`). Place starts with the block already in the gripper.
- **Gates:** each on 500 test episodes, teacher ≥ 90% and student ≥ 90%.
- **Accounting:** record the control steps spent training each skill
  (teacher PPO steps plus student DAgger rollout steps), for the spec's
  "skill steps count" variant.

## Review Focus

1. **Place starts that fail to hold the block,** or spawn it inside another
   block. Expected: after a reset the gripper holds the block, and nothing
   overlaps. Test: Task 4, `test_place_start_holds_the_block`.
2. **Ending the skill early without succeeding.** Expected: the episode ends
   with the penalty, success is 0, and the world restarts. Test: Task 4,
   `test_stopping_without_success_is_penalised`.
3. **The timeout.** Expected: `done = 1` and `truncation = 1` at step 150.
   Test: Task 4, `test_timeout_truncates`.
4. **Success detection that a real solution never triggers,** which would
   make the 90% gates unreachable. Expected: a scripted controller succeeds
   in at least 80% of worlds. Test: Task 4, `test_scripted_pick_succeeds`
   and `test_scripted_place_succeeds`.
5. **Saved policies.** Expected: a teacher or student loaded from disk acts
   exactly like the one that was saved. Test: Task 5
   `test_saved_teacher_reloads`; Task 7 `test_saved_student_reloads`.

## File Structure

- Modify `environments/stack_sim.py`: `Sim(..., cameras=True)`, so
  teachers can skip the render context.
- Modify `environments/stack_control.py`: `Controller.solve(..., iters=None)`.
- Modify `environments/stack.py`: factor `arm_features(sim, control, d)` out
  of `BatchStack._advance_impl` and reuse it.
- Create `skills/__init__.py` (empty), `skills/brax_util.py`,
  `skills/tasks.py`, `skills/env.py`, `skills/scripted.py`,
  `skills/teacher.py`, `skills/evaluate.py`, `skills/student.py`,
  `skills/policy.py`, `skills/README.md`.
- Create tests: `tests/test_brax_ppo.py`, `tests/test_skill_tasks.py`,
  `tests/test_skill_env.py`, `tests/test_skill_teacher.py`,
  `tests/test_skill_student.py`.
- Outputs: `outputs/train/skill_{pick,place}_{teacher,student}/`
  (`params.pkl`, `config.json`, `metrics.jsonl`) and
  `outputs/eval/skill_{pick,place}/{teacher,student}_gate.json`.

---

### Task 1: Install Brax and check its PPO trains a batched environment

This task is a gate. If PPO cannot train an environment that resets its own
worlds, stop and report.

**Files:**
- Create: `skills/__init__.py`, `skills/brax_util.py`, `tests/test_brax_ppo.py`

**Interfaces:**
- Produces, in `skills/brax_util.py`:
  - `episode_info(batch, metric_names) -> dict`, with `truncation`,
    `episode_done` and `episode_metrics` (`sum_reward`, `length`, and one
    entry per metric), all float32 zeros of shape (batch,)
  - `track_episode(info, reward, done, truncation, metrics) -> dict`: the
    same bookkeeping as Brax's `EpisodeWrapper`, for environments that
    reset their own worlds (run with `ppo.train(..., wrap_env=False)`).

- [ ] **Step 1: Install**

```bash
cd /home/max/PlanningWithWorldModels
PY=~/miniconda3/envs/dreamerv3/bin/python
$PY -m pip install -c constraints.txt brax==0.14.2
$PY -c "import jax, brax, flax; print(jax.__version__, brax.__version__, flax.__version__)"
$PY -m pytest tests -q -x -k "not throughput" 2>&1 | tail -1
```

Expected: `0.10.2 0.14.2 0.12.8`, and the stage 1 suite still passes.

- [ ] **Step 2: Write the failing test**

`tests/test_brax_ppo.py`:

```python
"""Brax PPO trains a batched environment that resets its own worlds."""

import jax
import jax.numpy as jnp
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

from skills.brax_util import episode_info, track_episode

LENGTH = 20


class Point(Env):
  """Worlds move a point by 0.1 * action; reward is -|x|. Each world
  restarts at a random x after LENGTH steps."""

  def __init__(self, batch):
    self.batch = batch

  def reset(self, rng):
    x = jax.random.uniform(rng[0], (self.batch,), minval=-1, maxval=1)
    info = dict(episode_info(self.batch, ['size']), rng=rng[0],
                steps=jnp.zeros(self.batch))
    return State(None, {'state': x[:, None]}, jnp.zeros(self.batch),
                 jnp.zeros(self.batch), {'size': jnp.zeros(self.batch)}, info)

  def step(self, state, action):
    x = state.obs['state'][:, 0] + 0.1 * action[:, 0]
    reward = -jnp.abs(x)
    steps = state.info['steps'] + 1
    done = (steps >= LENGTH).astype(jnp.float32)
    info = track_episode(state.info, reward, done, done,
                         {'size': jnp.abs(x)})
    rng, key = jax.random.split(info['rng'])
    fresh = jax.random.uniform(key, (self.batch,), minval=-1, maxval=1)
    x = jnp.where(done > 0, fresh, x)
    info = dict(info, rng=rng, steps=jnp.where(done > 0, 0.0, steps))
    return state.replace(obs={'state': x[:, None]}, reward=reward,
                         done=done, metrics={'size': jnp.abs(x)}, info=info)

  @property
  def observation_size(self):
    return {'state': (1,)}

  @property
  def action_size(self):
    return 1

  @property
  def backend(self):
    return 'none'


def test_ppo_learns_a_self_resetting_batched_env():
  rewards = []
  ppo.train(
      environment=Point(256), eval_env=Point(64), wrap_env=False,
      num_envs=256, num_eval_envs=64, episode_length=LENGTH,
      num_timesteps=300_000, num_evals=3, batch_size=8,
      num_minibatches=32, unroll_length=10, learning_rate=3e-3,
      network_factory=_networks(),
      progress_fn=lambda step, m: rewards.append(
          float(m['eval/episode_reward'])))
  assert rewards[-1] > rewards[0] + 2.0, rewards


def _networks():
  import functools
  from brax.training.agents.ppo import networks
  return functools.partial(
      networks.make_ppo_networks, policy_hidden_layer_sizes=(32, 32),
      value_hidden_layer_sizes=(32, 32), policy_obs_key='state',
      value_obs_key='state')
```

- [ ] **Step 3: Run it to see it fail**

Run: `$PY -m pytest tests/test_brax_ppo.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills'`.

- [ ] **Step 4: Write the bookkeeping helpers**

`skills/__init__.py`: an empty file.

`skills/brax_util.py`:

```python
"""Episode bookkeeping for Brax PPO with environments that reset their own
worlds.

Brax's EpisodeWrapper and AutoResetWrapper keep these fields, but its
auto-reset returns every world to its first start. The skill environments
draw a new start whenever a world's episode ends, so they skip Brax's
wrappers (ppo.train(..., wrap_env=False)) and keep the same fields here.
"""

import jax.numpy as jnp


def episode_info(batch, metric_names):
  """The fields Brax PPO reads from state.info, for a fresh batch."""
  zeros = jnp.zeros(batch, jnp.float32)
  metrics = {'sum_reward': zeros, 'length': zeros}
  metrics.update({name: zeros for name in metric_names})
  return {'truncation': zeros, 'episode_done': zeros,
          'episode_metrics': metrics}


def track_episode(info, reward, done, truncation, metrics):
  """Adds one step to each world's episode totals, as EpisodeWrapper does;
  totals restart on the step after a world's episode ended."""
  keep = 1.0 - info['episode_done']
  totals = {k: v * keep for k, v in info['episode_metrics'].items()}
  totals['sum_reward'] = totals['sum_reward'] + reward
  totals['length'] = totals['length'] + 1.0
  for name, value in metrics.items():
    totals[name] = totals[name] + value
  return dict(info, truncation=truncation.astype(jnp.float32),
              episode_done=done.astype(jnp.float32), episode_metrics=totals)
```

- [ ] **Step 5: Run the test**

Run: `$PY -m pytest tests/test_brax_ppo.py -q -s`
Expected: `1 passed`. Eval reward rises from about −10 to above −8. If PPO
raises about a missing `info` key, add that key to `episode_info` with the
shape Brax expects (read `brax/envs/wrappers/training.py`) and record it in
the ledger.

- [ ] **Step 6: Commit**

```bash
git add skills/__init__.py skills/brax_util.py tests/test_brax_ppo.py
git commit -m "Add Brax PPO bookkeeping for self-resetting batched environments"
```

---

### Task 2: Small extensions to the stage 1 environment

**Files:**
- Modify: `environments/stack_sim.py`, `environments/stack_control.py`,
  `environments/stack.py`
- Test: `tests/test_stack_sim.py`, `tests/test_stack_control.py`

**Interfaces:**
- Produces:
  - `Sim(num_envs, size=(64, 64), cameras=True)`: with `cameras=False` no
    render context is made, and `render` raises `RuntimeError`.
  - `Controller.solve(qpos, pos, mat, iters=None)`: `iters` overrides
    `self.iters`.
  - `environments.stack.arm_features(sim, control, d) -> (B, 19) float32`.
    `BatchStack` uses it for its `arm` observation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stack_sim.py`:

```python
def test_sim_without_cameras_steps_but_does_not_render():
  sim = Sim(8, cameras=False)
  d = jax.jit(sim.physics, static_argnums=2)(
      sim.make_data(), home_ctrl(8), 10)
  assert np.isfinite(np.asarray(jax.device_get(d.qpos))).all()
  with pytest.raises(RuntimeError):
    sim.render(d)
```

Append to `tests/test_stack_control.py`:

```python
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
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_sim.py tests/test_stack_control.py -q -k "without_cameras or far_target"`
Expected: FAIL with `TypeError: Sim.__init__() got an unexpected keyword
argument 'cameras'` and `TypeError: ... unexpected keyword argument 'iters'`.

- [ ] **Step 3: Implement**

In `environments/stack_sim.py`, give `Sim.__init__` a `cameras=True`
argument, make the render context only when it is set, and guard `render`:

```python
  def __init__(self, num_envs, size=(64, 64), cameras=True):
```

```python
    self.context = None
    if cameras:
      self.context = warp_io.create_render_context(
          mjm=m, nworld=num_envs, cam_res=(self.size[1], self.size[0]),
          use_textures=True, use_shadows=False, render_rgb=True,
          render_depth=True, enabled_geom_groups=[0, 1, 2])
```

```python
  def render(self, d):
    """Both cameras for every world, as uint8 RGB."""
    if self.context is None:
      raise RuntimeError('This Sim was made with cameras=False.')
    ctx = self.context.pytree()
```

In `environments/stack_control.py`:

```python
  def solve(self, qpos, pos, mat, iters=None):
    """Arm joint angles that put the gripper site at `pos` with
    orientation `mat`, for one world."""
```

and in its body
`return jax.lax.fori_loop(0, iters or self.iters, body, qpos[:7])`.

In `environments/stack.py`, add at module level:

```python
def arm_features(sim, control, d):
  """The arm observation: joint angles and velocities, gripper position,
  yaw and opening."""
  ee_pos, ee_yaw = control.ee(d)
  return jnp.concatenate([
      d.qpos[:, :7], d.qvel[:, :7], ee_pos, ee_yaw[:, None],
      sim.opening(d)[:, None]], -1).astype(jnp.float32)
```

and in `BatchStack._advance_impl` replace the `ee_pos, ee_yaw = ...` and
`arm = jnp.concatenate(...)` lines with
`arm = arm_features(self.sim, self.control, d)`, and the observation's
`'arm': arm.astype(jnp.float32)` with `'arm': arm`.

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_stack_sim.py tests/test_stack_control.py tests/test_stack_env.py -q`
Expected: all pass (the 2 new ones plus the existing ones).

- [ ] **Step 5: Commit**

```bash
git add environments/stack_sim.py environments/stack_control.py environments/stack.py tests/test_stack_sim.py tests/test_stack_control.py
git commit -m "Let Sim skip cameras, IK take more iterations, and share arm features"
```

---

### Task 3: Pick and place task logic

**Files:**
- Create: `skills/tasks.py`
- Test: `tests/test_skill_tasks.py`

**Interfaces:**
- Consumes: `sample_start`, `NONE`, `XY_TOL`, `Z_TOL` (`stack_task`);
  `COLORS`, `HALF`, `OPEN_WIDTH` (`stack_scene`).
- Produces, in `skills/tasks.py` (single-world JAX functions):
  - **Constants:** `EPISODE = 150`, `LIFT = 0.05`, `RAISE = 0.05`,
    `TABLE = 4`, `SUCCESS_BONUS = 5.0`, `FAIL_PENALTY = 1.0`,
    `SITE_LO = (0.40, -0.15, 0.17)`, `SITE_HI = (0.60, 0.15, 0.22)`.
  - `clear_mask(pos (4, 3), present (4,)) -> (4,) bool`: blocks with nothing
    on top of them.
  - `clearance(pos, present, c) -> float`: the gap under block c's bottom.
  - `resting_on(pos, h, t) -> bool`, and
    `place_goal(pos, h, t) -> (3,)`: where block h's centre goes.
  - `pick_progress(pos, present, held, site, c) -> (phi, success)`.
  - `place_progress(pos, held, site, opening, h, t) -> (phi, success)`.
  - `pick_start(key, prestack) -> (present, pos, yaws, target)`.
  - `place_start(key, prestack) -> (present, pos, yaws, held_block, target, site, yaw)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skill_tasks.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np

from environments.stack_scene import HALF, PARK
from skills.tasks import (
    LIFT, SITE_HI, SITE_LO, TABLE, clear_mask, clearance, pick_progress,
    pick_start, place_progress, place_start, resting_on)

ALL = jnp.ones(4, bool)
NOBODY = jnp.zeros(4, bool)


def layout(**blocks):
  """Positions with the named colours (r, g, b, y) placed, others parked."""
  pos = PARK.copy()
  for name, xyz in blocks.items():
    pos['rgby'.index(name)] = xyz
  return jnp.asarray(pos)


def test_clear_blocks_have_nothing_on_top():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF), b=(0.4, 0.1, HALF))
  present = jnp.array([True, True, True, False])
  assert clear_mask(pos, present).tolist() == [False, True, True, False]


def test_clearance_measures_the_gap_below():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF + 0.03))
  present = jnp.array([True, True, False, False])
  assert abs(float(clearance(pos, present, 1)) - 0.03) < 1e-5
  assert abs(float(clearance(pos, present, 0))) < 1e-6


def test_pick_succeeds_only_when_held_and_lifted():
  pos = layout(r=(0.5, 0, HALF + LIFT + 0.01))
  present = jnp.array([True, False, False, False])
  site = pos[0]
  _, ok = pick_progress(pos, present, NOBODY.at[0].set(True), site, 0)
  assert bool(ok)
  _, ok = pick_progress(pos, present, NOBODY, site, 0)
  assert not bool(ok)
  low = layout(r=(0.5, 0, HALF + 0.02))
  _, ok = pick_progress(low, present, NOBODY.at[0].set(True), low[0], 0)
  assert not bool(ok)


def test_pick_potential_grows_as_the_gripper_nears():
  pos = layout(r=(0.5, 0, HALF))
  present = jnp.array([True, False, False, False])
  far, _ = pick_progress(pos, present, NOBODY, jnp.array([0.5, 0, 0.3]), 0)
  near, _ = pick_progress(pos, present, NOBODY, jnp.array([0.5, 0, 0.1]), 0)
  assert float(near) > float(far)


def test_resting_on_a_block_and_on_the_table():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0.005, 3 * HALF))
  assert bool(resting_on(pos, 1, 0))
  assert not bool(resting_on(pos, 1, TABLE))
  assert bool(resting_on(pos, 0, TABLE))


def test_place_succeeds_when_released_open_and_raised():
  pos = layout(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF))
  top = 3 * HALF + HALF
  site = jnp.array([0.5, 0, top + 0.06])
  _, ok = place_progress(pos, NOBODY, site, 0.08, 1, 0)
  assert bool(ok)
  _, ok = place_progress(pos, NOBODY.at[1].set(True), site, 0.04, 1, 0)
  assert not bool(ok)
  _, ok = place_progress(pos, NOBODY, site.at[2].set(top + 0.02), 0.08, 1, 0)
  assert not bool(ok)


def test_pick_targets_are_present_and_clear():
  keys = jax.random.split(jax.random.PRNGKey(0), 512)
  present, pos, _, target = jax.vmap(pick_start, (0, None))(keys, 0.5)
  for i in range(len(keys)):
    assert bool(clear_mask(pos[i], present[i])[target[i]])


def test_place_starts_put_the_block_in_reach_above_everything():
  keys = jax.random.split(jax.random.PRNGKey(1), 512)
  present, pos, _, held, target, site, _ = jax.vmap(
      place_start, (0, None))(keys, 0.5)
  pos, held, target, site = map(np.asarray, (pos, held, target, site))
  for i in range(len(keys)):
    np.testing.assert_allclose(pos[i, held[i]], site[i])
    assert (site[i] >= np.array(SITE_LO) - 1e-6).all()
    assert (site[i] <= np.array(SITE_HI) + 1e-6).all()
    others = [c for c in range(4) if present[i, c] and c != held[i]]
    assert pos[i, held[i], 2] - HALF > max(pos[i, others, 2] + HALF)
    assert target[i] != held[i]
    if target[i] != TABLE:
      assert present[i, target[i]]
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skill_tasks.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills.tasks'`.

- [ ] **Step 3: Write the task logic**

`skills/tasks.py`:

```python
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
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_skill_tasks.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add skills/tasks.py tests/test_skill_tasks.py
git commit -m "Add pick and place task logic for the skill teachers"
```

---

### Task 4: The skill environment and a scripted check

**Files:**
- Create: `skills/env.py`, `skills/scripted.py`
- Test: `tests/test_skill_env.py`

**Interfaces:**
- Consumes: Tasks 1–3; `Sim`, `Controller`, `rot_z`, `HOLD` and
  `arm_features` from stage 1; `held_mask` (`stack_task`).
- Produces, in `skills/env.py`:
  - **Constants:** `STATE_DIM = 85`, `PROPRIO_DIM = 25`, `ACTION_DIM = 6`,
    `SETTLE = 10`, `METRICS = ('success', 'stopped')`.
  - `class SkillEnv(brax.envs.base.Env)`:
    `SkillEnv(skill, num_envs, cameras=False, prestack=0.3)`, with `skill`
    either `'pick'` or `'place'`. Its `reset(rng (B, 2))` and
    `step(state, action (B, 6))` return a `brax.envs.base.State`.
    - `obs['state']` is (B, 85): arm (19); 4 blocks × [present, position
      relative to the gripper site (3), position (3), quaternion (4),
      velocity (3)]; target one-hot (5); held one-hot (4); time fraction (1).
    - With `cameras=True` it adds `obs['workspace']`, `obs['wrist']` (uint8
      (B, 64, 64, 3)) and `obs['proprio']` (B, 25: arm, target one-hot,
      time fraction).
    - Metrics are `success` (1 on a step that ends the skill having
      succeeded) and `stopped` (1 when the policy ended the skill).
    - `info` has Task 1's Brax fields plus `rng`, `task` (`present` (B, 4),
      `target` (B,), `held_block` (B,)), `ee_target`, `ee_yaw`, `phi` and
      `steps`.
    - A world whose episode ended restarts within the same step, as Brax
      expects: the returned `obs` is its new start, while `reward` and `done`
      belong to the step that ended.
    - `held(state) -> (B, 4) bool`.
- Produces, in `skills/scripted.py`: `pick(state_obs) -> (B, 6)` and
  `place(state_obs) -> (B, 6)`: controllers that read only `obs['state']`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skill_env.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from skills import scripted
from skills.env import ACTION_DIM, PROPRIO_DIM, STATE_DIM, SkillEnv
from skills.tasks import EPISODE, FAIL_PENALTY


def start(skill, batch, cameras=False, seed=0):
  env = SkillEnv(skill, batch, cameras=cameras)
  keys = jax.random.split(jax.random.PRNGKey(seed), batch)
  return env, jax.jit(env.reset)(keys), jax.jit(env.step)


def run(env, state, step, policy, steps=EPISODE):
  """Steps until every world has ended once; first-episode success."""
  ended = np.zeros(env.num_envs, bool)
  success = np.zeros(env.num_envs, bool)
  for _ in range(steps):
    state = step(state, policy(state.obs['state']))
    done = np.asarray(state.done) > 0
    first = done & ~ended
    success[first] = np.asarray(state.metrics['success'])[first] > 0
    ended |= done
    if ended.all():
      break
  return success, ended


@pytest.mark.parametrize('skill', ['pick', 'place'])
def test_reset_shapes(skill):
  env, state, _ = start(skill, 8)
  assert state.obs['state'].shape == (8, STATE_DIM)
  assert env.action_size == ACTION_DIM
  assert np.isfinite(np.asarray(state.obs['state'])).all()


def test_cameras_add_student_inputs():
  _, state, _ = start('pick', 4, cameras=True)
  assert state.obs['workspace'].shape == (4, 64, 64, 3)
  assert state.obs['workspace'].dtype == jnp.uint8
  assert state.obs['proprio'].shape == (4, PROPRIO_DIM)


def test_place_start_holds_the_block():
  env, state, _ = start('place', 64)
  held = np.asarray(env.held(state))
  block = np.asarray(state.info['task']['held_block'])
  assert held[np.arange(64), block].mean() >= 0.95


def test_stopping_without_success_is_penalised():
  env, state, step = start('pick', 8)
  stop = jnp.zeros((8, ACTION_DIM)).at[:, 5].set(1.0)
  state = step(state, stop)
  assert (np.asarray(state.done) == 1).all()
  assert (np.asarray(state.metrics['success']) == 0).all()
  assert (np.asarray(state.reward) < -FAIL_PENALTY + 0.2).all()
  assert (np.asarray(state.info['steps']) == 0).all()  # Restarted.


def test_timeout_truncates():
  env, state, step = start('pick', 4)
  hold = jnp.zeros((4, ACTION_DIM)).at[:, 4:].set(-1.0)
  for _ in range(EPISODE):
    state = step(state, hold)
  assert (np.asarray(state.done) == 1).all()
  assert (np.asarray(state.info['truncation']) == 1).all()


def test_scripted_pick_succeeds():
  env, state, step = start('pick', 64, seed=1)
  success, ended = run(env, state, step, scripted.pick)
  assert ended.all()
  assert success.mean() >= 0.8, success.mean()


def test_scripted_place_succeeds():
  env, state, step = start('place', 64, seed=2)
  success, ended = run(env, state, step, scripted.place)
  assert ended.all()
  assert success.mean() >= 0.8, success.mean()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skill_env.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills.env'`.

- [ ] **Step 3: Write the environment**

`skills/env.py`:

```python
"""Batched Brax environments for the pick and place skills.

Each world starts from the stacking start distribution: for pick, the arm
at home and a clear block to pick; for place, a block already held above
the others and a target for it (a clear block or the table). The policy
moves the arm through the raw 5-D action and ends the skill by setting
action[5] > 0. A world whose episode ended restarts within the same step,
with its own new start (Brax's AutoResetWrapper would replay its first one),
so PPO runs with wrap_env=False and the bookkeeping in brax_util.
"""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State

from environments import stack_scene as scene
from environments.stack import arm_features
from environments.stack_control import Controller, rot_z
from environments.stack_sim import Sim
from environments.stack_task import held_mask
from skills import tasks
from skills.brax_util import episode_info, track_episode

STATE_DIM = 85
PROPRIO_DIM = 25
ACTION_DIM = 6
SETTLE = 10  # Physics steps after a new start.
METRICS = ('success', 'stopped')


class SkillEnv(Env):

  def __init__(self, skill, num_envs, cameras=False, prestack=0.3,
               size=(64, 64)):
    assert skill in ('pick', 'place'), skill
    self.skill, self.num_envs, self.cameras = skill, num_envs, cameras
    self.prestack = prestack
    self.sim = Sim(num_envs, size, cameras=cameras)
    self.control = Controller(self.sim)

  @property
  def observation_size(self):
    size = {'state': (STATE_DIM,)}
    if self.cameras:
      image = self.sim.size + (3,)
      size.update(workspace=image, wrist=image, proprio=(PROPRIO_DIM,))
    return size

  @property
  def action_size(self):
    return ACTION_DIM

  @property
  def backend(self):
    return 'mjx'

  def reset(self, rng):
    d = self.sim.make_data()
    d, task = self._fresh(d, rng)
    target, yaw = self.control.start(d)
    phi, _ = self._progress(d, task)
    zeros = jnp.zeros(self.num_envs)
    info = dict(episode_info(self.num_envs, METRICS), rng=rng, task=task,
                ee_target=target, ee_yaw=yaw, phi=phi, steps=zeros)
    obs, d = self._obs(d, info)
    metrics = {name: zeros for name in METRICS}
    return State(d, obs, zeros, zeros, metrics, info)

  def step(self, state, action):
    info = state.info
    a = jnp.clip(action, -1.0, 1.0)
    d = state.pipeline_state
    ctrl, target, yaw = self.control.act(
        d, info['ee_target'], info['ee_yaw'], a[:, :5])
    d = self.sim.physics(d, ctrl, scene.SUBSTEPS)
    steps = info['steps'] + 1
    phi, success = self._progress(d, info['task'])
    stopped = a[:, 5] > 0
    timeout = steps >= tasks.EPISODE
    bonus = jnp.where(success, tasks.SUCCESS_BONUS, -tasks.FAIL_PENALTY)
    reward = phi - info['phi'] + jnp.where(stopped, bonus, 0.0)
    done = stopped | timeout
    metrics = {'success': (stopped & success).astype(jnp.float32),
               'stopped': stopped.astype(jnp.float32)}
    info = track_episode(info, reward, done, timeout & ~stopped, metrics)
    info = dict(info, ee_target=target, ee_yaw=yaw, phi=phi, steps=steps)
    d, info = jax.lax.cond(
        done.any(), lambda: self._restart(d, info, done), lambda: (d, info))
    obs, d = self._obs(d, info)
    return State(d, obs, reward.astype(jnp.float32),
                 done.astype(jnp.float32), metrics, info)

  def held(self, state):
    d = state.pipeline_state
    return jax.vmap(held_mask)(
        self.sim.block_pos(d), self.sim.site_pos(d),
        self.sim.opening(d)) & state.info['task']['present']

  def _fresh(self, d, keys):
    """New starts for every world, settled."""
    d = self.sim.home(d)
    if self.skill == 'pick':
      present, pos, yaws, target = jax.vmap(tasks.pick_start, (0, None))(
          keys, self.prestack)
      held_block = target
    else:
      present, pos, yaws, held_block, target, site, yaw = jax.vmap(
          tasks.place_start, (0, None))(keys, self.prestack)
      mats = jax.vmap(rot_z)(yaw) @ self.control.home_mat
      q = jax.vmap(lambda qpos, p, m: self.control.solve(
          qpos, p, m, iters=30))(d.qpos, site, mats)
      fingers = jnp.full((self.num_envs, 2), scene.HALF)
      d = d.replace(
          qpos=d.qpos.at[:, :7].set(q).at[:, 7:9].set(fingers),
          ctrl=d.ctrl.at[:, :7].set(q).at[:, 7].set(scene.CLOSED))
    d = self.sim.set_blocks(d, pos, yaws)
    d = self.sim.physics(d, d.ctrl, SETTLE)
    task = {'present': present, 'target': target.astype(jnp.int32),
            'held_block': held_block.astype(jnp.int32)}
    return d, task

  def _restart(self, d, info, mask):
    """New starts for the worlds in `mask`; the others are untouched."""
    split = jax.vmap(jax.random.split)(info['rng'])
    rng, keys = split[:, 0], split[:, 1]
    fresh, task = self._fresh(d, keys)
    m = mask[:, None]
    d = self.sim.forward(d.replace(
        qpos=jnp.where(m, fresh.qpos, d.qpos),
        qvel=jnp.where(m, fresh.qvel, d.qvel),
        ctrl=jnp.where(m, fresh.ctrl, d.ctrl)))

    def pick(new, old):
      return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)),
                       new, old)

    task = jax.tree.map(pick, task, info['task'])
    target, yaw = self.control.start(d)
    phi, _ = self._progress(d, task)
    return d, dict(
        info, rng=rng, task=task,
        ee_target=pick(target, info['ee_target']),
        ee_yaw=pick(yaw, info['ee_yaw']), phi=pick(phi, info['phi']),
        steps=jnp.where(mask, 0.0, info['steps']))

  def _progress(self, d, task):
    pos, site = self.sim.block_pos(d), self.sim.site_pos(d)
    opening = self.sim.opening(d)
    held = jax.vmap(held_mask)(pos, site, opening) & task['present']
    if self.skill == 'pick':
      return jax.vmap(tasks.pick_progress)(
          pos, task['present'], held, site, task['target'])
    return jax.vmap(tasks.place_progress)(
        pos, held, site, opening, task['held_block'], task['target'])

  def _obs(self, d, info):
    task = info['task']
    arm = arm_features(self.sim, self.control, d)
    pos, site = self.sim.block_pos(d), self.sim.site_pos(d)
    present = task['present'][..., None].astype(jnp.float32)
    blocks = jnp.concatenate([
        present, pos - site[:, None], pos, self.sim.block_quat(d),
        self.sim.block_vel(d)], -1) * present
    held = jax.vmap(held_mask)(pos, site, self.sim.opening(d))
    held = (held & task['present']).astype(jnp.float32)
    target = jax.nn.one_hot(task['target'], 5)
    time = (info['steps'] / tasks.EPISODE)[:, None]
    obs = {'state': jnp.concatenate([
        arm, blocks.reshape(self.num_envs, -1), target, held, time], -1)}
    if self.cameras:
      workspace, wrist, d = self.sim.render(d)
      obs.update(workspace=workspace, wrist=wrist,
                 proprio=jnp.concatenate([arm, target, time], -1))
    return obs, d
```

- [ ] **Step 4: Write the scripted controllers**

`skills/scripted.py`:

```python
"""Scripted pick and place from privileged state, used to check that each
skill environment can be solved and that its success is detected. They read
only obs['state'] (layout in skills/env.py)."""

import jax.numpy as jnp
import numpy as np

from environments.stack_control import MAX_STEP, MAX_YAW
from environments.stack_scene import HALF

ALIGN = 0.008  # How close in xy before descending, metres.
YAW_OK = 0.05


def _split(state):
  b = state.shape[0]
  blocks = state[:, 19:75].reshape(b, 4, 14)
  return dict(
      ee=state[:, 14:17], yaw=state[:, 17], opening=state[:, 18],
      present=blocks[..., 0] > 0, pos=blocks[..., 4:7],
      quat=blocks[..., 7:11], target=jnp.argmax(state[:, 75:80], -1),
      held=state[:, 80:84] > 0)


def _block_yaw(quat):
  w, x, y, z = (quat[..., i] for i in range(4))
  return jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _turn(ee_yaw, block_yaw):
  """Yaw step that lines the fingers up with the nearest pair of faces."""
  err = (block_yaw - ee_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
  return jnp.clip(err / MAX_YAW, -1, 1), jnp.abs(err) < YAW_OK


def _act(ee, want, turn, grip, stop):
  move = jnp.clip((want - ee) / MAX_STEP, -1, 1)
  return jnp.concatenate([move, turn[:, None], grip[:, None],
                          stop[:, None]], -1)


def pick(state):
  s = _split(state)
  rows = jnp.arange(state.shape[0])
  p = s['pos'][rows, s['target']]
  held = s['held'][rows, s['target']]
  turn, aligned = _turn(s['yaw'], _block_yaw(s['quat'][rows, s['target']]))
  xy_ok = jnp.linalg.norm(p[:, :2] - s['ee'][:, :2], axis=-1) < ALIGN
  above = p + jnp.array([0.0, 0.0, 0.08])
  want = jnp.where((xy_ok & aligned)[:, None], p, above)
  near = jnp.linalg.norm(s['ee'] - p, axis=-1) < ALIGN
  lifted = held & (p[:, 2] >= 0.21)
  want = jnp.where(held[:, None], s['ee'] + jnp.array([0.0, 0.0, 0.03]),
                   want)
  grip = jnp.where(held | near, 1.0, -1.0)
  return _act(s['ee'], want, turn, grip, jnp.where(lifted, 1.0, -1.0))


def place(state):
  s = _split(state)
  rows = jnp.arange(state.shape[0])
  holding = s['held'].any(-1)
  nearest = jnp.argmin(jnp.where(
      s['present'], jnp.linalg.norm(
          s['pos'][..., :2] - s['ee'][:, None, :2], axis=-1), 9.0), -1)
  h = jnp.where(holding, jnp.argmax(s['held'], -1), nearest)
  p = s['pos'][rows, h]
  t = s['target']
  on_block = s['pos'][rows, jnp.minimum(t, 3)] + jnp.array([0, 0, 2 * HALF])
  table = jnp.stack([p[:, 0], p[:, 1], jnp.full_like(p[:, 0], HALF)], -1)
  goal = jnp.where((t == 4)[:, None], table, on_block)
  goal_yaw = jnp.where(t == 4, _block_yaw(s['quat'][rows, h]),
                       _block_yaw(s['quat'][rows, jnp.minimum(t, 3)]))
  turn, _ = _turn(_block_yaw(s['quat'][rows, h]), goal_yaw)
  xy_ok = jnp.linalg.norm(goal[:, :2] - p[:, :2], axis=-1) < 0.005
  safe = jnp.maximum(p[:, 2], goal[:, 2] + 0.04)
  want_block = jnp.where(
      xy_ok[:, None], goal,
      jnp.stack([goal[:, 0], goal[:, 1], safe], -1))
  down = xy_ok & (jnp.abs(p[:, 2] - goal[:, 2]) < 0.004)
  want = s['ee'] + (want_block - p)
  want = jnp.where(holding[:, None], want,
                   s['ee'] + jnp.array([0.0, 0.0, 0.03]))
  grip = jnp.where(holding & ~down, 1.0, -1.0)
  raised = s['ee'][:, 2] - (p[:, 2] + HALF) >= 0.055
  stop = ~holding & raised & (s['opening'] > 0.07)
  turn = jnp.where(holding, turn, 0.0)
  return _act(s['ee'], want, turn, grip, jnp.where(stop, 1.0, -1.0))
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_skill_env.py -q`
Expected: `8 passed`.
- **If a scripted test is below 80%:** find out whether the environment or
  the script is at fault before changing either. Save the failing worlds'
  wrist frames (rerun with `cameras=True`) and print their `obs['state']`
  fields over time. A script that visibly succeeds while `success` stays 0
  is an environment bug: fix `skills/tasks.py`, with a test. A script that
  fumbles is a script bug: tune `ALIGN`, the heights, or the order of its
  phases.
- **If `jax.lax.cond` around `_restart` fails to compile with MuJoCo Warp
  calls inside:** compute `_restart` every step and select with
  `jnp.where` instead, and record the ruling (it costs speed).

- [ ] **Step 6: Commit**

```bash
git add skills/env.py skills/scripted.py tests/test_skill_env.py
git commit -m "Add batched pick and place environments with a scripted check"
```

---

### Task 5: Teacher training, loading and the success gate

**Files:**
- Create: `skills/teacher.py`, `skills/evaluate.py`, `skills/policy.py`
- Test: `tests/test_skill_teacher.py`

**Interfaces:**
- Consumes: `SkillEnv` (Task 4).
- Produces:
  - `skills/teacher.py`:
    - `TEACHER = dict(policy=(256, 256, 256), value=(256,) * 5)`.
    - `PPO = dict(...)` with the defaults below.
    - `train(skill, out, num_envs=8192, timesteps=100_000_000, seed=0, **ppo_overrides) -> dict`
      writes `out/params.pkl`, `out/config.json` and `out/metrics.jsonl`, and
      returns the config (which includes `control_steps`).
    - CLI: `python skills/teacher.py --skill pick`.
  - `skills/policy.py`:
    - `load_teacher(path) -> fn(state_obs (B, 85)) -> (B, 6)`, deterministic
      and jittable.
    - `load_student(path)` (written in Task 7).
  - `skills/evaluate.py`:
    - `success_rate(skill, policy, episodes=500, cameras=False, seed=0) -> float`:
      the first episode of each of `episodes` worlds. `policy` takes the
      full `obs` dict.
    - CLI: `python skills/evaluate.py --skill pick --who teacher` writes
      `outputs/eval/skill_<skill>/<who>_gate.json`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skill_teacher.py`:

```python
import json

import jax
import numpy as np

from skills import scripted
from skills.evaluate import success_rate
from skills.policy import load_teacher
from skills.teacher import train


def test_scripted_policy_passes_through_the_evaluator():
  rate = success_rate('pick', lambda obs: scripted.pick(obs['state']),
                      episodes=64)
  assert rate >= 0.8


def test_short_teacher_run_saves_a_policy(tmp_path):
  config = train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4,
                 batch_size=2, num_minibatches=32, num_evals=2,
                 num_eval_envs=16)
  assert (tmp_path / 'params.pkl').exists()
  assert json.loads((tmp_path / 'config.json').read_text())['skill'] == 'pick'
  assert config['control_steps'] >= 64 * 10 * 4
  assert (tmp_path / 'metrics.jsonl').read_text().strip()


def test_saved_teacher_reloads(tmp_path):
  train('pick', tmp_path, num_envs=64, timesteps=64 * 10 * 4, batch_size=2,
        num_minibatches=32, num_evals=2, num_eval_envs=16)
  policy = load_teacher(tmp_path)
  obs = jax.random.normal(jax.random.PRNGKey(0), (5, 85))
  a, b = np.asarray(policy(obs)), np.asarray(jax.jit(policy)(obs))
  assert a.shape == (5, 6) and np.abs(a).max() <= 1.0
  np.testing.assert_allclose(a, b, atol=1e-6)
  again = np.asarray(load_teacher(tmp_path)(obs))
  np.testing.assert_allclose(a, again, atol=1e-6)
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skill_teacher.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills.evaluate'`.

- [ ] **Step 3: Write the evaluator**

`skills/evaluate.py`:

```python
"""Success rate of a skill policy over fresh starts: the gates of stage 2.

  python skills/evaluate.py --skill pick --who teacher
  python skills/evaluate.py --skill place --who student --episodes 500

Plays one episode in each of --episodes worlds and writes
outputs/eval/skill_<skill>/<who>_gate.json.
"""

import argparse
import json
import pathlib

import jax
import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent


def success_rate(skill, policy, episodes=500, cameras=False, seed=0):
  """Share of worlds whose first episode ends in success."""
  from skills.env import SkillEnv
  from skills.tasks import EPISODE
  env = SkillEnv(skill, episodes, cameras=cameras)
  keys = jax.random.split(jax.random.PRNGKey(seed), episodes)
  state = jax.jit(env.reset)(keys)
  step = jax.jit(lambda s: env.step(s, policy(s.obs)))
  ended = np.zeros(episodes, bool)
  success = np.zeros(episodes, bool)
  for _ in range(EPISODE):
    state = step(state)
    done = np.asarray(state.done) > 0
    first = done & ~ended
    success[first] = np.asarray(state.metrics['success'])[first] > 0
    ended |= done
    if ended.all():
      break
  return float(success.mean())


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--who', choices=['teacher', 'student', 'scripted'],
                      required=True)
  parser.add_argument('--episodes', type=int, default=500)
  parser.add_argument('--seed', type=int, default=1)
  args = parser.parse_args()
  from skills import policy as policies
  from skills import scripted
  run = REPO / 'outputs' / 'train' / f'skill_{args.skill}_{args.who}'
  if args.who == 'teacher':
    act = policies.load_teacher(run)
    policy, cameras = (lambda obs: act(obs['state'])), False
  elif args.who == 'student':
    policy, cameras = policies.load_student(run), True
  else:
    script = getattr(scripted, args.skill)
    policy, cameras = (lambda obs: script(obs['state'])), False
  rate = success_rate(args.skill, policy, args.episodes, cameras, args.seed)
  out = REPO / 'outputs' / 'eval' / f'skill_{args.skill}'
  out.mkdir(parents=True, exist_ok=True)
  result = dict(skill=args.skill, who=args.who, episodes=args.episodes,
                seed=args.seed, success=rate, passed=rate >= 0.9)
  (out / f'{args.who}_gate.json').write_text(json.dumps(result, indent=2))
  print(json.dumps(result))


if __name__ == '__main__':
  main()
```

- [ ] **Step 4: Write teacher training and loading**

`skills/teacher.py`:

```python
"""Train a skill's state teacher with Brax PPO on the GPU.

  python skills/teacher.py --skill pick
  python skills/teacher.py --skill place --timesteps 200000000

Writes outputs/train/skill_<skill>_teacher/: params.pkl (normaliser, policy
and value parameters), config.json (settings and control steps used) and
metrics.jsonl (one line per evaluation).
"""

import argparse
import functools
import json
import pathlib
import pickle
import time

import jax

REPO = pathlib.Path(__file__).resolve().parent.parent

TEACHER = dict(policy=(256, 256, 256), value=(256,) * 5)
PPO = dict(
    episode_length=150, unroll_length=10, num_minibatches=32,
    num_updates_per_batch=8, discounting=0.97, learning_rate=1e-3,
    entropy_cost=1e-2, batch_size=256, num_evals=10, num_eval_envs=256,
    normalize_observations=True, reward_scaling=1.0)


def network_factory():
  from brax.training.agents.ppo import networks
  return functools.partial(
      networks.make_ppo_networks,
      policy_hidden_layer_sizes=TEACHER['policy'],
      value_hidden_layer_sizes=TEACHER['value'],
      policy_obs_key='state', value_obs_key='state')


def train(skill, out, num_envs=8192, timesteps=100_000_000, seed=0,
          **overrides):
  from brax.training.agents.ppo import train as ppo
  from skills.env import SkillEnv
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  settings = {**PPO, **overrides}
  log = (out / 'metrics.jsonl').open('w')
  last = {'step': 0}
  start = time.time()

  def progress(step, metrics):
    row = {'step': int(step), 'seconds': time.time() - start}
    row.update({k: float(v) for k, v in metrics.items()})
    last.update(row)
    log.write(json.dumps(row) + '\n')
    log.flush()
    print(f"{step:>12,}  success {row.get('eval/episode_success', 0):.3f}"
          f"  reward {row.get('eval/episode_reward', 0):.2f}", flush=True)

  _, params, _ = ppo.train(
      environment=SkillEnv(skill, num_envs),
      eval_env=SkillEnv(skill, settings['num_eval_envs']),
      wrap_env=False, num_envs=num_envs, num_timesteps=timesteps,
      network_factory=network_factory(), seed=seed, progress_fn=progress,
      **settings)
  with (out / 'params.pkl').open('wb') as f:
    pickle.dump(jax.device_get(params), f)
  config = dict(skill=skill, num_envs=num_envs, timesteps=timesteps,
                seed=seed, network=TEACHER, ppo=settings,
                control_steps=max(last['step'], timesteps),
                seconds=time.time() - start)
  (out / 'config.json').write_text(json.dumps(config, indent=2))
  return config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--timesteps', type=int, default=100_000_000)
  parser.add_argument('--num-envs', type=int, default=8192)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--entropy-cost', type=float, default=PPO['entropy_cost'])
  args = parser.parse_args()
  out = REPO / 'outputs' / 'train' / f'skill_{args.skill}_teacher'
  config = train(args.skill, out, args.num_envs, args.timesteps, args.seed,
                 entropy_cost=args.entropy_cost)
  print(f"Wrote {out}: {config['control_steps']:,} control steps in "
        f"{config['seconds'] / 60:.0f} min")


if __name__ == '__main__':
  main()
```

`skills/policy.py`:

```python
"""Trained skills as JAX functions, loaded from their run directories."""

import json
import pathlib
import pickle

import jax


def load_teacher(path):
  """The teacher's deterministic policy: state obs (B, 85) to actions
  (B, 6)."""
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks
  from skills.env import ACTION_DIM, STATE_DIM
  from skills.teacher import network_factory
  path = pathlib.Path(path)
  params = pickle.loads((path / 'params.pkl').read_bytes())
  network = network_factory()(
      {'state': (STATE_DIM,)}, ACTION_DIM,
      preprocess_observations_fn=running_statistics.normalize)
  act = networks.make_inference_fn(network)(params, deterministic=True)
  key = jax.random.PRNGKey(0)
  return lambda state: act({'state': state}, key)[0]


def config(path):
  return json.loads((pathlib.Path(path) / 'config.json').read_text())
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest tests/test_skill_teacher.py -q`
Expected: `3 passed`. If `ppo.train` rejects the tiny settings (Brax
requires `batch_size * num_minibatches` to be a multiple of `num_envs`),
adjust the test's numbers, not the defaults.

- [ ] **Step 6: Commit**

```bash
git add skills/teacher.py skills/evaluate.py skills/policy.py tests/test_skill_teacher.py
git commit -m "Train skill teachers with Brax PPO and gate them on success"
```

---

### Task 6: Train the pick and place teachers (run)

**Files:**
- Create: `outputs/train/skill_pick_teacher/`, `outputs/train/skill_place_teacher/`,
  `outputs/eval/skill_pick/teacher_gate.json`, `outputs/eval/skill_place/teacher_gate.json`,
  `outputs/logs/skill_{pick,place}_teacher.log`

**Interfaces:**
- Consumes: `skills/teacher.py`, `skills/evaluate.py` (Task 5).
- Produces: teacher parameters that pass the 90% gate, for Task 7.

- [ ] **Step 1: Check the scripted baseline on 500 episodes**

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
$PY skills/evaluate.py --skill pick --who scripted
$PY skills/evaluate.py --skill place --who scripted
```

Expected: success ≥ 0.8 for each. This bounds what the environment allows,
and the teachers should reach it.

- [ ] **Step 2: Train the pick teacher**

```bash
$PY skills/teacher.py --skill pick > outputs/logs/skill_pick_teacher.log 2>&1
tail -3 outputs/logs/skill_pick_teacher.log
```

Expected: about 1–1.5 h. The log's evaluation success climbs over the 10
evaluations. Write the wall-clock time into the ledger.

- [ ] **Step 3: Gate the pick teacher**

Run: `$PY skills/evaluate.py --skill pick --who teacher`
Expected: `"passed": true` (success ≥ 0.9). If it falls short, retry at
most twice, recording each attempt in the ledger:
1. `--timesteps 200000000`;
2. then `--entropy-cost 0.005`, at 200M.

If it is still below 0.9, **stop and report**: the success rates, the
evaluation curve, and wrist frames of failures. Possible next steps are
reward-weight changes or using the scripted controller as the teacher, and
that choice is the user's.

- [ ] **Step 4: Train and gate the place teacher**

```bash
$PY skills/teacher.py --skill place > outputs/logs/skill_place_teacher.log 2>&1
$PY skills/evaluate.py --skill place --who teacher
```

The same expectation, retries and stop rule as for pick.

- [ ] **Step 5: Commit the results**

```bash
git add outputs/train/skill_pick_teacher outputs/train/skill_place_teacher \
  outputs/eval/skill_pick outputs/eval/skill_place outputs/logs/skill_*_teacher.log
git commit -m "Train the pick and place teachers" -m "<success rates, steps and times>"
```

---

### Task 7: Image students by DAgger

**Files:**
- Create: `skills/student.py`
- Modify: `skills/policy.py` (add `load_student`)
- Test: `tests/test_skill_student.py`

**Interfaces:**
- Consumes: `SkillEnv(..., cameras=True)` (Task 4), `load_teacher`
  (Task 5), `scripted` (Task 4, in tests).
- Produces, in `skills/student.py`:
  - `class Student(flax.linen.Module)`: `__call__(images uint8
    (B, 64, 64, 6), proprio (B, 25)) -> (B, 6)`. Five tanh outputs, then
    one terminate logit.
  - `act(params, obs) -> (B, 6)`: actions in [-1, 1]. The terminate output
    is +1 when the logit is above 0, else -1.
  - `dagger(skill, teacher, out, iterations=20, num_envs=128, rollout=300,
    stride=2, train_steps=2000, capacity=150_000, batch=256, seed=0) -> dict`:
    writes `out/params.pkl`, `out/config.json` (including `control_steps`)
    and `out/metrics.jsonl`. `teacher` is a function from state obs to
    actions.
  - CLI: `python skills/student.py --skill pick` (loads
    `outputs/train/skill_<skill>_teacher`).
- Produces, in `skills/policy.py`: `load_student(path) -> fn(obs dict) -> (B, 6)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_skill_student.py`:

```python
import json

import jax
import jax.numpy as jnp
import numpy as np

from skills import scripted
from skills.policy import load_student
from skills.student import Student, act, dagger


def test_student_outputs_actions_in_range():
  net = Student()
  images = jnp.zeros((3, 64, 64, 6), jnp.uint8)
  proprio = jnp.zeros((3, 25))
  params = net.init(jax.random.PRNGKey(0), images, proprio)
  out = act(params, {'workspace': images[..., :3], 'wrist': images[..., 3:],
                     'proprio': proprio})
  assert out.shape == (3, 6)
  assert np.abs(np.asarray(out)).max() <= 1.0
  assert set(np.unique(np.asarray(out[:, 5])).tolist()) <= {-1.0, 1.0}


def test_dagger_fits_the_teacher(tmp_path):
  config = dagger('pick', scripted.pick, tmp_path, iterations=3, num_envs=8,
                  rollout=40, stride=2, train_steps=150, capacity=2000,
                  batch=64)
  losses = [json.loads(line)['loss']
            for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
  assert losses[-1] < losses[0]
  assert config['control_steps'] == 3 * 8 * 40
  assert (tmp_path / 'params.pkl').exists()


def test_saved_student_reloads(tmp_path):
  dagger('pick', scripted.pick, tmp_path, iterations=1, num_envs=4,
         rollout=10, stride=1, train_steps=5, capacity=100, batch=8)
  policy = load_student(tmp_path)
  obs = {'workspace': jnp.zeros((2, 64, 64, 3), jnp.uint8),
         'wrist': jnp.full((2, 64, 64, 3), 128, jnp.uint8),
         'proprio': jnp.ones((2, 25))}
  np.testing.assert_allclose(np.asarray(policy(obs)),
                             np.asarray(load_student(tmp_path)(obs)))
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skill_student.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'skills.student'`.

- [ ] **Step 3: Write the student and DAgger**

`skills/student.py`:

```python
"""Distil a skill's state teacher into an image student with DAgger.

  python skills/student.py --skill pick
  python skills/student.py --skill place --iterations 30

The student sees what the option environment will give it: both cameras,
the arm, the target and the time fraction. Each iteration plays the
batched skill environment with the student acting in each world (the
teacher acts with probability 0.5 ** iteration instead), labels every
visited state with the teacher's action, and trains the student on all
labels so far. Writes outputs/train/skill_<skill>_student/.
"""

import argparse
import json
import pathlib
import pickle
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO = pathlib.Path(__file__).resolve().parent.parent


class Student(nn.Module):
  """Both cameras through a small CNN, joined with the proprioception."""

  @nn.compact
  def __call__(self, images, proprio):
    x = images.astype(jnp.float32) / 255.0 - 0.5
    for depth in (32, 64, 128, 256):
      x = nn.relu(nn.Conv(depth, (4, 4), strides=(2, 2))(x))
    x = x.reshape(x.shape[0], -1)
    x = jnp.concatenate([x, proprio], -1)
    x = nn.relu(nn.Dense(512)(x))
    x = nn.relu(nn.Dense(256)(x))
    return nn.Dense(6)(x)


def _inputs(obs):
  images = jnp.concatenate([obs['workspace'], obs['wrist']], -1)
  return images, obs['proprio']


def act(params, obs):
  """Actions in [-1, 1]; the terminate output is +1 or -1."""
  out = Student().apply(params, *_inputs(obs))
  stop = jnp.where(out[:, 5] > 0, 1.0, -1.0)
  return jnp.concatenate([jnp.tanh(out[:, :5]), stop[:, None]], -1)


def _loss(params, images, proprio, target):
  out = Student().apply(params, images, proprio)
  move = jnp.mean((jnp.tanh(out[:, :5]) - target[:, :5]) ** 2)
  label = (target[:, 5] > 0).astype(jnp.float32)
  stop = jnp.mean(optax.sigmoid_binary_cross_entropy(out[:, 5], label))
  return move + stop


def dagger(skill, teacher, out, iterations=20, num_envs=128, rollout=300,
           stride=2, train_steps=2000, capacity=150_000, batch=256, seed=0):
  from skills.env import PROPRIO_DIM, SkillEnv
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  env = SkillEnv(skill, num_envs, cameras=True)
  key = jax.random.PRNGKey(seed)
  key, init_key, reset_key = jax.random.split(key, 3)
  state = jax.jit(env.reset)(jax.random.split(reset_key, num_envs))
  params = Student().init(init_key, *_inputs(state.obs))
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)

  @jax.jit
  def collect(state, params, key, beta):
    labels = teacher(state.obs['state'])
    mine = act(params, state.obs)
    use_teacher = jax.random.uniform(key, (num_envs,)) < beta
    action = jnp.where(use_teacher[:, None], labels, mine)
    return env.step(state, action), labels

  @jax.jit
  def update(params, opt_state, images, proprio, target):
    loss, grads = jax.value_and_grad(_loss)(params, images, proprio, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

  images = np.zeros((capacity, 64, 64, 6), np.uint8)
  proprio = np.zeros((capacity, PROPRIO_DIM), np.float32)
  targets = np.zeros((capacity, 6), np.float32)
  size, cursor, steps = 0, 0, 0
  rng = np.random.default_rng(seed)
  log = (out / 'metrics.jsonl').open('w')
  start = time.time()
  for it in range(iterations):
    beta = 0.5 ** it
    for t in range(rollout):
      key, sub = jax.random.split(key)
      obs = state.obs
      state, labels = collect(state, params, sub, beta)
      steps += num_envs
      if t % stride:
        continue
      ims, prop = jax.device_get(_inputs(obs))
      slots = (cursor + np.arange(num_envs)) % capacity
      images[slots], proprio[slots] = ims, prop
      targets[slots] = np.asarray(jax.device_get(labels))
      cursor = int((cursor + num_envs) % capacity)
      size = min(size + num_envs, capacity)
    losses = []
    for _ in range(train_steps):
      idx = rng.integers(0, size, batch)
      params, opt_state, loss = update(
          params, opt_state, jax.device_put(images[idx]),
          jax.device_put(proprio[idx]), jax.device_put(targets[idx]))
      losses.append(float(loss))
    row = dict(iteration=it, beta=beta, samples=size, control_steps=steps,
               loss=float(np.mean(losses[-50:])),
               seconds=time.time() - start)
    log.write(json.dumps(row) + '\n')
    log.flush()
    print(json.dumps(row), flush=True)
  with (out / 'params.pkl').open('wb') as f:
    pickle.dump(jax.device_get(params), f)
  config = dict(skill=skill, iterations=iterations, num_envs=num_envs,
                rollout=rollout, stride=stride, train_steps=train_steps,
                capacity=capacity, batch=batch, seed=seed,
                control_steps=steps, seconds=time.time() - start)
  (out / 'config.json').write_text(json.dumps(config, indent=2))
  return config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--iterations', type=int, default=20)
  parser.add_argument('--num-envs', type=int, default=128)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  from skills.policy import load_teacher
  runs = REPO / 'outputs' / 'train'
  teacher = load_teacher(runs / f'skill_{args.skill}_teacher')
  config = dagger(args.skill, teacher, runs / f'skill_{args.skill}_student',
                  args.iterations, args.num_envs, seed=args.seed)
  print(f"{config['control_steps']:,} control steps in "
        f"{config['seconds'] / 60:.0f} min")


if __name__ == '__main__':
  main()
```

Add to `skills/policy.py`:

```python
def load_student(path):
  """The image student's policy: obs dict with workspace, wrist and
  proprio to actions (B, 6)."""
  from skills import student
  params = pickle.loads((pathlib.Path(path) / 'params.pkl').read_bytes())
  return lambda obs: student.act(params, obs)
```

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_skill_student.py -q`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add skills/student.py skills/policy.py tests/test_skill_student.py
git commit -m "Distil skill teachers into image students with DAgger"
```

---

### Task 8: Train the students (run) and document stage 2

**Files:**
- Create: `outputs/train/skill_{pick,place}_student/`,
  `outputs/eval/skill_{pick,place}/student_gate.json`,
  `outputs/logs/skill_{pick,place}_student.log`, `skills/README.md`
- Modify: `AGENTS.md` (Layout)

**Interfaces:**
- Consumes: Tasks 5–7.
- Produces: students that pass the gate, loadable with
  `skills.policy.load_student`, for stage 4; and the stage's step and time
  accounting.

- [ ] **Step 1: Train and gate the pick student**

```bash
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # must be empty
$PY skills/student.py --skill pick > outputs/logs/skill_pick_student.log 2>&1
$PY skills/evaluate.py --skill pick --who student
```

Expected: the DAgger loss falls over the iterations, and
`"passed": true` (success ≥ 0.9). If it falls short, retry at most twice:
1. `--iterations 30`;
2. then `--num-envs 256`.

If it is still below, **stop and report**, with the teacher's and student's
success rates and student failure frames. The likely gap is perception at
64×64, and changing the resolution is the user's call.

- [ ] **Step 2: Train and gate the place student**

```bash
$PY skills/student.py --skill place > outputs/logs/skill_place_student.log 2>&1
$PY skills/evaluate.py --skill place --who student
```

The same expectation, retries and stop rule.

- [ ] **Step 3: Write `skills/README.md`**

```markdown
# Skills

The options of the block-stacking experiment: `pick(colour)` and
`place_on(colour | table)`, each a policy on the raw end-effector action
plus a terminate output (action[5] > 0 ends the skill).

    python skills/evaluate.py --skill pick --who scripted   # environment check
    python skills/teacher.py --skill pick                   # state teacher, Brax PPO
    python skills/evaluate.py --skill pick --who teacher    # gate: >= 90% of 500
    python skills/student.py --skill pick                   # image student, DAgger
    python skills/evaluate.py --skill pick --who student    # gate: >= 90% of 500

- `env.py`: batched Brax environments on the stage 1 simulator; worlds
  restart on their own with new starts, so PPO runs with `wrap_env=False`.
- `tasks.py`: starts, success and the shaped potentials teachers learn from.
- `teacher.py`, `student.py`, `evaluate.py`, `policy.py`: training, gates,
  and loading skills as JAX functions (`load_student` for stage 4).
- `scripted.py`: scripted controllers that check each task is solvable.

Runs are in `outputs/train/skill_<skill>_{teacher,student}/`; each
`config.json` records `control_steps`, the control steps spent training it.
```

Add one line about `skills/` to the Layout section of `AGENTS.md`:

```markdown
- `skills/`: the pick and place skills behind the block-stacking options
  (Brax PPO teachers distilled into image students); see `skills/README.md`.
```

- [ ] **Step 4: Record the accounting and timing**

Read `control_steps` and `seconds` from the four `config.json` files and
put them in the commit message, with the total. Compare the wall-clock time
with "Where to run" above. If any run took more than 3 h, say in the final
report whether Delta or DeltaAI would have paid off.

- [ ] **Step 5: Commit**

```bash
git add outputs/train/skill_pick_student outputs/train/skill_place_student \
  outputs/eval/skill_pick outputs/eval/skill_place \
  outputs/logs/skill_*_student.log skills/README.md AGENTS.md
git commit -m "Train the pick and place students and document the skills" \
  -m "<gate results; control steps per run and total; wall-clock times>"
```
