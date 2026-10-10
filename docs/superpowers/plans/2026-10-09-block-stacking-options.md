# Block Stacking Option Agent Implementation Plan (Stage 4 of 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The option environment `custom_stack_options` (9 discrete options,
each running a trained image skill), checked by tests and a scripted N = 3
tower, and a DreamerV3 N = 2 pilot on it at the raw pilot's control-step
budget.

**Architecture:** `environments/stack_options.py` wraps stage 1's
`BatchStack`. One option step runs a Python loop of jitted control ticks: in
each tick every world whose option is still running acts with its skill (the
pick or place student, on device), and worlds whose option has ended are
frozen (their `qpos`, `qvel` and `ctrl` restored), until every world's option
has ended. The tick reuses `BatchStack._advance_impl`, so reward, tower
prefix, success and the time limit are exactly the raw agent's. Before that,
the skills are made to chain: pick students only ever started with the arm
at home, place students with the block held 17–22 cm up, and the table
placement put the block straight down, onto whatever was under it.

**Tech Stack:** JAX 0.10.2, MJX Warp (stage 1 simulator), Flax students
(stage 2), DreamerV3 at `e3f0224` through `train.py`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-07-block-stacking-options-design.md`
(sections "Option action space", "Skills", "Agents and training"; stage 4
of "Build order").

## Global Constraints

- Code style from AGENTS.md: 2-space indentation, single quotes, about 80
  columns, no type hints, short docstrings, comments describe the code as
  it is.
- Do not edit `third_party/`.
- `PY=~/miniconda3/envs/dreamerv3/bin/python`. Tests: `$PY -m pytest tests/...`
  from the repository root.
- One GPU process at a time: `nvidia-smi` must list no compute process
  before GPU tests, training or evaluation.
- Physics and rendering go through `environments/stack_sim.py`.
- Spec values: 9 options (`pick(c)` for 4 colours, `place_on(c)` for 4
  colours, `place_on(table)`); an invalid option does nothing for 10 control
  steps; a skill runs until its success condition holds or for at most 150
  control steps; an option that would cross the episode's time limit is cut
  off at it; the option's reward is the sum over its control steps; the
  option environment adds `log/duration`; `privileged()` adds
  `option_mask`.
- Validity (spec): `pick(c)` with an empty gripper and c present and clear;
  `place_on(c)` while holding a block other than c, with c present and
  clear; `place_on(table)` while holding a block.
- Student gate: at least 90% of 500 episodes (`skills/evaluate.py`).
- DreamerV3 settings equal to the raw pilot `stack_raw_n2`: default size
  (`deter` 2048), `train_ratio` 32, `contdisc`, `run.envs` 16,
  `--jax.prealloc False`, `--replay.size` as in its config, and
  `MALLOC_MMAP_THRESHOLD_=1048576`.

## Review Focus

1. **Frozen worlds stay frozen.** A world whose option has ended must not
   move or gain control steps while other worlds' options run. Test:
   `test_worlds_wait_for_each_other` (Task 3).
2. **An invalid option while holding a block does not drop it.** "Doing
   nothing" keeps the gripper closed if it is closed. Test:
   `test_invalid_option_keeps_the_grip` (Task 3).
3. **The finishing place waits for success.** After the last block is
   placed the episode needs 1 s with the gripper open; the option holds
   still until success is decided, rather than handing control back while
   the tower is unconfirmed. Test: `test_completed_tower_waits_for_success`
   (Task 3).
4. **The time limit cuts an option off.** Test:
   `test_time_limit_cuts_the_option` (Task 3).
5. **DreamerV3's transfer guard.** Every step moves data explicitly. Test:
   `test_runs_under_transfer_guard` (Task 3).

---

### Task 1: Skills that chain

The pick student has only seen the arm at home; inside an option it starts
wherever the last place left it (raised 5 cm over a block). The place
student has only held the block 17–22 cm up; after a pick it is 5 cm clear
of where it was. And `place_on(table)` aims straight down, onto a block if
the block was picked off a stack. Pick starts gain an arm pose over the
blocks (half the time; the rest at home), place starts hold the block from
just clear of the blocks under it up to 22 cm, and the table placement
aims at the nearest free cell of the start grid.

**Files:**
- Modify: `skills/tasks.py`, `skills/env.py:105-130` (`_fresh`),
  `skills/scripted.py:62-90` (`place`)
- Test: `tests/test_skill_tasks.py`, `tests/test_skill_env.py`

**Interfaces:**
- Produces: `tasks.table_spot(pos, present, h) -> (3,)`;
  `tasks.arm_start(key, pos, present) -> (site (3,), yaw ())`;
  `tasks.place_goal(pos, present, h, t)` and
  `tasks.place_progress(pos, present, held, site, opening, h, t)` (new
  `present` argument); `SkillEnv(..., home=tasks.HOME_SHARE)`.

- [ ] **Step 1: Write the failing tests** in `tests/test_skill_tasks.py`
  (add `table_spot, arm_start, FREE, RAISE, ARM_LO, ARM_HI` to the import
  from `skills.tasks`):

```python
def test_table_spot_is_free_and_on_the_table():
  keys = jax.random.split(jax.random.PRNGKey(5), 256)
  present, pos, _, held, _, _, _ = jax.vmap(
      place_start, (0, None))(keys, 1.0)
  spots = jax.vmap(table_spot)(pos, present, held)
  pos, present, held, spots = map(np.asarray, (pos, present, held, spots))
  for i in range(len(keys)):
    assert spots[i, 2] == pytest.approx(HALF)
    for c in range(4):
      if present[i, c] and c != held[i]:
        assert np.linalg.norm(spots[i, :2] - pos[i, c, :2]) >= FREE - 1e-6


def test_table_spot_is_under_the_block_when_that_is_free():
  pos = layout(r=(0.45, -0.05, 0.15), g=(0.65, 0.15, HALF))
  present = jnp.array([True, True, False, False])
  spot = np.asarray(table_spot(pos, present, 0))
  np.testing.assert_allclose(spot, [0.45, -0.05, HALF], atol=1e-6)


def test_arm_starts_are_clear_of_the_blocks():
  keys = jax.random.split(jax.random.PRNGKey(6), 512)
  present, pos, _, _ = jax.vmap(pick_start, (0, None))(keys, 1.0)
  site, yaw = jax.vmap(arm_start)(keys, pos, present)
  pos, present, site = map(np.asarray, (pos, present, site))
  assert (site[:, :2] >= np.array(ARM_LO) - 1e-6).all()
  assert (site[:, :2] <= np.array(ARM_HI) + 1e-6).all()
  assert (np.abs(np.asarray(yaw)) <= np.pi / 4 + 1e-6).all()
  for i in range(len(keys)):
    near = present[i] & (np.linalg.norm(
        pos[i, :, :2] - site[i, :2], axis=-1) < FREE)
    top = max([0.0] + list(pos[i, near, 2] + HALF))
    assert site[i, 2] >= top + RAISE - 1e-6
```

  Replace `test_place_starts_put_the_block_in_reach_above_everything`
  with:

```python
def test_place_starts_hold_the_block_clear_of_the_blocks_under_it():
  keys = jax.random.split(jax.random.PRNGKey(1), 512)
  present, pos, _, held, target, site, _ = jax.vmap(
      place_start, (0, None))(keys, 0.5)
  pos, held, target, site = map(np.asarray, (pos, held, target, site))
  low = 0
  for i in range(len(keys)):
    np.testing.assert_allclose(pos[i, held[i]], site[i])
    assert (site[i, :2] >= np.array(ARM_LO) - 1e-6).all()
    assert (site[i, :2] <= np.array(ARM_HI) + 1e-6).all()
    assert site[i, 2] <= SITE_HI[2] + 1e-6
    others = [c for c in range(4) if present[i, c] and c != held[i]
              and np.linalg.norm(pos[i, c, :2] - site[i, :2]) < FREE]
    under = max([0.0] + [pos[i, c, 2] + HALF for c in others])
    assert site[i, 2] - HALF >= under + 0.01 - 1e-6
    low += site[i, 2] < 0.12
    assert target[i] != held[i]
    if target[i] != TABLE:
      assert present[i, target[i]]
  assert low > 50  # Some starts hold the block low, as after a pick.
```

  Update the three `place_progress(pos, NOBODY, ...)` calls in
  `test_place_success_*` to pass `present` second:
  `place_progress(pos, PRESENT, held, site, opening, 1, 0)` with
  `PRESENT = jnp.array([True, True, False, False])` defined beside
  `NOBODY`.

  In `tests/test_skill_env.py` add:

```python
def test_pick_starts_put_the_arm_over_the_blocks_or_at_home():
  env = SkillEnv('pick', 64, home=0.5)
  state = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(2), 64))
  site = np.asarray(env.sim.site_pos(state.pipeline_state))
  home = np.asarray(env.sim.site_pos(env.sim.forward(
      env.sim.home(state.pipeline_state))))
  at_home = np.linalg.norm(site - home, axis=-1) < 0.01
  assert 10 < at_home.sum() < 54
  assert (np.asarray(env.sim.opening(state.pipeline_state)) >
          scene.OPEN_WIDTH).all()
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_skill_tasks.py tests/test_skill_env.py -q`
Expected: ImportError for `table_spot`.

- [ ] **Step 3: Implement** in `skills/tasks.py`. Import `SLOTS` from
  `environments.stack_task`. Constants:

```python
FREE = 0.06  # Table cells closer than this to another block are taken.
ARM_LO = (0.35, -0.18)  # Gripper xy range of chained starts.
ARM_HI = (0.65, 0.18)
HOME_SHARE = 0.5  # Share of pick starts with the arm at home.
_SLOTS = jnp.asarray(SLOTS)
```

  Replace `SITE_LO` with `SITE_HI = (0.65, 0.18, 0.22)` only (place starts
  draw xy from `ARM_LO`/`ARM_HI`). New functions:

```python
def top_under(pos, present, xy):
  """The highest top of the present blocks within FREE of xy, or 0."""
  near = (jnp.linalg.norm(pos[:, :2] - xy, axis=-1) < FREE) & present
  return jnp.max(jnp.where(near, pos[:, 2] + HALF, 0.0))


def table_spot(pos, present, h):
  """Where block h goes on the table: the start-grid cell nearest it that
  is at least FREE from every other present block."""
  others = present & (_IDX != h)
  dist = jnp.linalg.norm(_SLOTS[:, None] - pos[None, :, :2], axis=-1)
  free = jnp.where(others[None], dist >= FREE, True).all(1)
  near = jnp.linalg.norm(_SLOTS - pos[h, :2], axis=-1)
  # The block's own spot counts when it is free, so it is a candidate too.
  own_free = jnp.where(others, jnp.linalg.norm(
      pos[:, :2] - pos[h, :2], axis=-1) >= FREE, True).all()
  cell = _SLOTS[jnp.argmin(jnp.where(free, near, jnp.inf))]
  xy = jnp.where(own_free, pos[h, :2], cell)
  return jnp.concatenate([xy, jnp.array([HALF], pos.dtype)])


def arm_start(key, pos, present):
  """A gripper site and yaw over the workspace, RAISE or more above the
  blocks under it, as a place leaves the arm."""
  k1, k2, k3 = jax.random.split(key, 3)
  xy = jax.random.uniform(k1, (2,), minval=jnp.array(ARM_LO),
                          maxval=jnp.array(ARM_HI))
  z = top_under(pos, present, xy) + RAISE
  z = jnp.maximum(z, 2 * HALF + RAISE) + jax.random.uniform(k2, (), maxval=0.15)
  yaw = jax.random.uniform(k3, (), minval=-np.pi / 4, maxval=np.pi / 4)
  return jnp.concatenate([xy, z[None]]), yaw
```

  `place_goal(pos, present, h, t)`: the table case returns
  `table_spot(pos, present, h)`. `place_progress(pos, present, held, site,
  opening, h, t)` passes `present` through. In `place_start`, replace the
  site draw:

```python
  rest = present & (_IDX != held)
  xy = jax.random.uniform(keys[2], (2,), minval=jnp.array(ARM_LO),
                          maxval=jnp.array(ARM_HI))
  low = top_under(pos, rest, xy) + HALF + 0.01
  z = jax.random.uniform(jax.random.fold_in(keys[2], 1), (), minval=low,
                         maxval=jnp.maximum(low, SITE_HI[2]))
  site = jnp.concatenate([xy, z[None]])
```

  The table-spot test `..._under_the_block_when_that_is_free`
  needs `layout` from the existing tests' helpers.

  In `skills/env.py`: `SkillEnv.__init__(..., home=tasks.HOME_SHARE)`
  stores `self.home`. `_fresh` for pick:

```python
      present, pos, yaws, target = jax.vmap(
          tasks.pick_start, (0, None, None))(keys, self.prestack, self.stacked)
      arm_keys = jax.vmap(lambda k: jax.random.fold_in(k, 7))(keys)
      site, yaw = jax.vmap(tasks.arm_start)(arm_keys, pos, present)
      at_home = jax.vmap(lambda k: jax.random.uniform(
          jax.random.fold_in(k, 8)) < self.home)(keys)
      mats = jax.vmap(rot_z)(yaw) @ self.control.home_mat
      q = jax.vmap(lambda qpos, p, m: self.control.solve(
          qpos, p, m, iters=30))(d.qpos, site, mats)
      q = jnp.where(at_home[:, None], d.qpos[:, :7], q)
      d = d.replace(qpos=d.qpos.at[:, :7].set(q),
                    ctrl=d.ctrl.at[:, :7].set(q))
      held_block = target
```

  and `_progress` passes `task['present']` to `place_progress`. In
  `skills/scripted.py` `place`, the table goal becomes the free spot:

```python
  table = jax.vmap(tasks.table_spot)(s['pos'], s['present'], h)
```

  (import `from skills import tasks`).

- [ ] **Step 4: Run the tests**

Run: `$PY -m pytest tests/test_skill_tasks.py tests/test_skill_env.py tests/test_skill_student.py tests/test_skill_teacher.py -q`
Expected: all pass.

- [ ] **Step 5: Scripted gates on the new starts** (GPU)

Run: `$PY skills/evaluate.py --skill pick --who scripted` and
`$PY skills/evaluate.py --skill place --who scripted`
Expected: success ≥ 0.95 each (were 1.0 and 0.966). If one falls below,
fix the scripted controller before going on; the students learn from it.

- [ ] **Step 6: Commit**

```bash
git add skills tests/test_skill_tasks.py tests/test_skill_env.py outputs/eval/skill_pick/scripted_gate.json outputs/eval/skill_place/scripted_gate.json
git commit -m "Start skills where other skills leave off, and place on a free spot"
```

### Task 2: Retrain and gate the students

**Files:**
- Move: `outputs/train/skill_{pick,place}_student` →
  `outputs/train/skill_{pick,place}_student_try2`, and
  `outputs/eval/skill_{pick,place}/student_gate.json` →
  `student_try2_gate.json` (the convention of the `_try1` runs).
- Modify: `skills/README.md` (rates), spec "Skills" paragraph (rates).

- [ ] **Step 1:** `git mv` the two run directories and the two gate files
  as above.
- [ ] **Step 2:** `$PY skills/student.py --skill pick 2>&1 | tee outputs/logs/skill_pick_student.log`
  then the same for `place` (about 8 min each).
- [ ] **Step 3:** `$PY skills/evaluate.py --skill pick --who student`, and
  for `place`. Expected: `passed: true` for both. If one fails, run 45
  DAgger iterations (`--iterations 45`) before anything else.
- [ ] **Step 4:** Update the rates in `skills/README.md` and the spec, and
  commit the runs, logs and gates:

```bash
git add outputs/train/skill_*_student* outputs/eval/skill_* outputs/logs/skill_*_student.log skills/README.md docs/superpowers/specs
git commit -m "Retrain the pick and place students on chained starts"
```

### Task 3: The option environment

**Files:**
- Create: `environments/stack_options.py`
- Modify: `environments/__init__.py` (`REGISTRY['stack_options']`),
  `environments/README.md` (a section on it)
- Test: `tests/test_stack_options.py`

**Interfaces:**
- Consumes: `BatchStack` (`_restart`, `_advance_impl`, `_privileged`,
  `sim`, `state`), `skills.policy.student_params`,
  `skills.student.act(params, obs)`, `skills.env.student_proprio(arm,
  target, steps)`, `tasks.clear_mask`, `tasks.pick_progress`,
  `tasks.place_progress(pos, present, held, site, opening, h, t)`.
- Produces: `NUM_OPTIONS = 9`, `INVALID_STEPS = 10`,
  `option_mask(pos, present, held) -> (9,) bool`,
  `BatchStackOptions(num_envs, size=(64, 64), n=0, prestack=0.3, seed=0,
  skills=None)` with `step(acts)` (`acts['action']` int (B,), `acts['reset']`
  bool (B,)), `privileged()` (adds `option_mask`), `.state`, `.base`;
  `StackOptions` (`custom_stack_options`).

- [ ] **Step 1: Write the failing tests** `tests/test_stack_options.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import environments
from environments.stack_options import (
    INVALID_STEPS, NUM_OPTIONS, BatchStackOptions, obs_space, option_mask)
from environments.stack_scene import HALF, PARK
from environments.stack_task import LIMITS

B = 4
TABLE = 8
NOBODY = jnp.zeros(4, bool)


def pos(**blocks):
  """Block positions: absent blocks parked, the named ones at (x, y, z)."""
  out = np.array(PARK, np.float32)
  for k, xyz in blocks.items():
    out['rgby'.index(k)] = xyz
  return jnp.asarray(out)


def acts(option, reset=False, batch=B):
  return {'reset': np.full(batch, reset),
          'action': np.broadcast_to(np.asarray(option, np.int32), (batch,))}


def teleport(env, world, placements):
  """Move blocks of one world: {colour: (x, y, z)}; others stay put."""
  sim, d = env.base.sim, env.state['data']
  p = np.array(jax.device_get(sim.block_pos(d)))
  for c, xyz in placements.items():
    p[world, c] = xyz
  env.state['data'] = sim.set_blocks(
      d, jax.device_put(p.astype(np.float32)),
      jax.device_put(np.zeros((env.num_envs, 4), np.float32)))


def test_mask_with_an_empty_gripper():
  present = jnp.array([True, True, False, False])
  m = np.asarray(option_mask(pos(r=(0.4, 0, HALF), g=(0.6, 0, HALF)),
                             present, NOBODY))
  assert m.tolist() == [1, 1, 0, 0] + [0] * 4 + [0]


def test_mask_while_holding():
  present = jnp.array([True, True, True, False])
  p = pos(r=(0.5, 0, 0.15), g=(0.4, 0, HALF), b=(0.6, 0, HALF))
  m = np.asarray(option_mask(p, present, NOBODY.at[0].set(True)))
  assert m.tolist() == [0] * 4 + [0, 1, 1, 0] + [1]


def test_mask_skips_covered_blocks():
  present = jnp.array([True, True, False, False])
  p = pos(r=(0.5, 0, HALF), g=(0.5, 0, 3 * HALF))
  m = np.asarray(option_mask(p, present, NOBODY))
  assert m[:2].tolist() == [0, 1]


def test_a_held_block_does_not_cover_the_one_below():
  present = jnp.array([True, True, False, False])
  p = pos(r=(0.5, 0, HALF), g=(0.5, 0, 0.12))
  m = np.asarray(option_mask(p, present, NOBODY.at[1].set(True)))
  assert m[4] and not m[5]


@pytest.fixture(scope='module')
def env():
  return BatchStackOptions(B, n=2, prestack=0.0, seed=0)


def check_obs(obs, batch=B):
  for key, space in obs_space((64, 64)).items():
    value = np.asarray(obs[key])
    assert value.shape == (batch,) + space.shape, key
    assert value.dtype == space.dtype, key


def test_first_step(env):
  obs = env.step(acts(TABLE, reset=True))
  check_obs(obs)
  assert obs['is_first'].all() and not obs['is_last'].any()
  assert (obs['reward'] == 0).all() and (obs['log/duration'] == 0).all()
  assert env.privileged()['option_mask'].shape == (B, NUM_OPTIONS)


def test_invalid_option_waits(env):
  first = env.step(acts(TABLE, reset=True))
  before = env.privileged()['blocks']
  obs = env.step(acts(TABLE))
  assert (obs['log/duration'] == INVALID_STEPS).all()
  assert (obs['log/sim_steps'] - first['log/sim_steps'] ==
          INVALID_STEPS).all()
  np.testing.assert_allclose(obs['arm'][:, 14:17], first['arm'][:, 14:17],
                             atol=0.01)
  np.testing.assert_allclose(env.privileged()['blocks'][..., 1:4],
                             before[..., 1:4], atol=0.005)
  assert (obs['reward'] == 0).all()


def test_pick_lifts_the_block(env):
  obs = env.step(acts(TABLE, reset=True))
  colour = obs['goal'][:, 0]
  obs = env.step({'reset': np.zeros(B, bool),
                  'action': colour.astype(np.int32)})
  held = env.privileged()['held']
  assert (held == colour).sum() >= 3
  assert (obs['log/duration'] < 150).sum() >= 3
  assert (obs['log/duration'] > INVALID_STEPS).all()


def test_worlds_wait_for_each_other(env):
  """Odd worlds run an invalid option while even worlds pick: the odd ones
  take exactly INVALID_STEPS and do not move, however long the picks run."""
  first = env.step(acts(TABLE, reset=True))
  before = env.privileged()['blocks']
  waits = np.arange(B) % 2 == 1
  options = np.where(waits, TABLE, first['goal'][:, 0]).astype(np.int32)
  obs = env.step({'reset': np.zeros(B, bool), 'action': options})
  delta = obs['log/sim_steps'] - first['log/sim_steps']
  assert (delta[waits] == INVALID_STEPS).all()
  assert (delta[~waits] > INVALID_STEPS).all()
  np.testing.assert_allclose(env.privileged()['blocks'][waits][..., 1:4],
                             before[waits][..., 1:4], atol=0.005)


def test_invalid_option_keeps_the_grip(env):
  obs = env.step(acts(TABLE, reset=True))
  colour = obs['goal'][:, 0]
  env.step({'reset': np.zeros(B, bool), 'action': colour.astype(np.int32)})
  held = env.privileged()['held']
  env.step(acts(colour[0]))  # pick(...) while holding: invalid.
  after = env.privileged()['held']
  assert ((held >= 0) <= (after == held)).all()


def test_completed_tower_waits_for_success(env):
  obs = env.step(acts(TABLE, reset=True))
  goal = obs['goal'][0]
  teleport(env, 0, {int(goal[0]): (0.5, 0.15, HALF),
                    int(goal[1]): (0.5, 0.15, 3 * HALF)})
  out = [env.step(acts(TABLE)) for _ in range(3)]
  assert [o['reward'][0] for o in out] == [2, 0, 1]
  assert [o['log/duration'][0] for o in out] == [10, 10, 1]
  assert out[2]['is_terminal'][0] and out[2]['is_last'][0]


def test_time_limit_cuts_the_option(env):
  env.step(acts(TABLE, reset=True))
  steps = np.array(jax.device_get(env.state['steps']))
  steps[:] = LIMITS[2] - 3
  env.state['steps'] = jax.device_put(steps.astype(np.int32))
  obs = env.step(acts(TABLE))
  assert (obs['log/duration'] == 3).all() and obs['is_last'].all()
  assert not obs['is_terminal'].any()


def test_runs_under_transfer_guard(env):
  with jax.transfer_guard('disallow'):
    env.step(acts(TABLE, reset=True))
    env.step(acts(TABLE))
    env.privileged()


def test_registered():
  e = environments.make('stack_options', batch=2, n=2)
  assert e.act_space['action'].discrete
  assert e.act_space['action'].high == NUM_OPTIONS
  assert 'log/duration' in e.obs_space
```

- [ ] **Step 2: Run them to see them fail**

Run: `$PY -m pytest tests/test_stack_options.py -q`
Expected: ImportError for `environments.stack_options`.

- [ ] **Step 3: Implement** `environments/stack_options.py`:

```python
"""Block stacking through options: pick(colour), place_on(colour | table).

BatchStackOptions wraps BatchStack. Options 0-3 pick a colour, 4-7 place
the held block on a colour and 8 on the table. A valid option runs its
trained image skill on the raw action space until the skill's success
condition holds, for at most SKILL_STEPS control steps; an invalid one holds
still for INVALID_STEPS. A place that completes the tower holds still until
the episode's success is decided. Every option ends when the episode does.
Worlds whose option has ended are frozen until every world's has. The
observation is the last control step's, with the reward summed over the
option and log/duration its control steps. StackOptions is the embodied.Env
(custom_stack_options).
"""

import pathlib

import elements
import jax
import jax.numpy as jnp
import numpy as np

from environments import stack
from environments import stack_scene as scene
from environments.stack_task import held_mask
from skills import tasks
from skills.env import student_proprio
from skills.policy import student_params
from skills.student import act as student_act

SKILLS = pathlib.Path(__file__).resolve().parent.parent / 'outputs' / 'train'
COLOURS = len(scene.COLORS)
NUM_OPTIONS = 2 * COLOURS + 1
SKILL_STEPS = tasks.EPISODE
INVALID_STEPS = 10
OBS = ('workspace', 'wrist', 'arm')


def option_mask(pos, present, held):
  """Which options are valid in one world."""
  holding = held.any()
  clear = tasks.clear_mask(pos, present & ~held)
  return jnp.concatenate([clear & ~holding, clear & holding, holding[None]])


def obs_space(size):
  return {**stack.obs_space(size),
          'log/duration': elements.Space(np.int32)}


def act_space():
  return {
      'reset': elements.Space(bool),
      'action': elements.Space(np.int32, (), 0, NUM_OPTIONS),
  }


def _pick(mask, new, old):
  return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old)


class BatchStackOptions:

  def __init__(self, num_envs, size=(64, 64), n=0, prestack=0.3, seed=0,
               skills=None):
    self.num_envs, self.size = num_envs, tuple(size)
    self.base = stack.BatchStack(num_envs, size, n, prestack, seed)
    skills = pathlib.Path(skills or SKILLS)
    b = num_envs
    with jax.transfer_guard('allow'):
      self.params = (student_params(skills / 'skill_pick_student'),
                     student_params(skills / 'skill_place_student'))
      self.obs = {k: jax.device_put(np.zeros(
          (b,) + v.shape, v.dtype)) for k, v in stack.obs_space(size).items()}
      option = jax.device_put(np.zeros(b, np.int32))
      mask = jax.device_put(np.ones(b, bool))
      self._begin = jax.jit(self._begin_impl).lower(
          self.base.state, option, mask).compile()
      opt = self._begin(self.base.state, option, mask)
      self._tick = jax.jit(self._tick_impl).lower(
          self.base.state, opt, self.obs, mask, self.params).compile()
      self._mask = jax.jit(self._mask_impl).lower(self.base.state).compile()
      self.no_first = jax.device_put(np.zeros(b, bool))
    self.done = np.ones(b, bool)

  @property
  def state(self):
    return self.base.state

  @state.setter
  def state(self, value):
    self.base.state = value

  def step(self, acts):
    """One option in every world; ended or flagged worlds restart and
    report their first observation instead."""
    restart = np.asarray(acts['reset'], bool) | self.done
    first = jax.device_put(restart)
    if restart.any():
      self.state = self.base._restart(self.state, first)
    option = jax.device_put(np.asarray(acts['action'], np.int32))
    opt = self._begin(self.state, option, first)
    self.state, opt, self.obs, running = self._tick(
        self.state, opt, self.obs, first, self.params)
    while bool(jax.device_get(running)):
      self.state, opt, self.obs, running = self._tick(
          self.state, opt, self.obs, self.no_first, self.params)
    out = dict(self.obs, reward=opt['reward'],
               **{'log/duration': opt['steps']})
    out = {k: np.asarray(v) for k, v in jax.device_get(out).items()}
    self.done = out['is_last'].copy()
    return out

  def privileged(self):
    out = dict(self.base.privileged())
    out['option_mask'] = np.asarray(jax.device_get(self._mask(self.state)))
    return out

  def _blocks(self, state):
    sim, d = self.base.sim, state['data']
    pos, site, opening = sim.block_pos(d), sim.site_pos(d), sim.opening(d)
    colours = jnp.arange(COLOURS)
    present = (colours[None, :, None] == state['goal'][:, None, :]).any(-1)
    held = jax.vmap(held_mask)(pos, site, opening) & present
    return pos, site, opening, present, held

  def _mask_impl(self, state):
    pos, _, _, present, held = self._blocks(state)
    return jax.vmap(option_mask)(pos, present, held)

  def _begin_impl(self, state, option, first):
    pos, _, _, present, held = self._blocks(state)
    mask = jax.vmap(option_mask)(pos, present, held)
    valid = jnp.take_along_axis(mask, option[:, None], 1)[:, 0]
    pick = option < COLOURS
    zeros = jnp.zeros(self.num_envs, jnp.int32)
    return dict(
        valid=valid, present=present, pick=pick,
        target=jnp.where(pick, option, option - COLOURS).astype(jnp.int32),
        held_block=jnp.argmax(held, -1).astype(jnp.int32),
        running=~first, skill_done=jnp.zeros(self.num_envs, bool),
        steps=zeros, reward=jnp.zeros(self.num_envs, jnp.float32))

  def _tick_impl(self, state, opt, obs, first, params):
    """One control step for the worlds whose option runs and the worlds
    starting an episode; the others are frozen."""
    d = state['data']
    student_obs = dict(
        workspace=obs['workspace'], wrist=obs['wrist'],
        proprio=student_proprio(obs['arm'], opt['target'], opt['steps']))
    skill = jnp.where(opt['pick'][:, None],
                      student_act(params[0], student_obs),
                      student_act(params[1], student_obs))
    # Holding still keeps the gripper as it is, so a held block stays held.
    closed = d.ctrl[:, 7] < scene.OPEN / 2
    still = jnp.zeros((self.num_envs, 5)).at[:, 4].set(
        jnp.where(closed, 1.0, -1.0))
    wait = ~opt['valid'] | opt['skill_done']
    action = jnp.where(wait[:, None], still, skill)
    new, new_obs = self.base._advance_impl(state, action, first)
    update = opt['running'] | first
    state = self._select(update, new, state)
    obs = {k: _pick(update, new_obs[k], obs[k]) for k in obs}

    ran = opt['running']
    steps = opt['steps'] + ran
    reward = opt['reward'] + jnp.where(ran, new_obs['reward'], 0.0)
    pos, site, opening, _, held = self._blocks(state)
    present, target = opt['present'], opt['target']
    _, picked = jax.vmap(tasks.pick_progress)(
        pos, present, held, site, jnp.minimum(target, COLOURS - 1))
    _, placed = jax.vmap(tasks.place_progress)(
        pos, present, held, site, opening, opt['held_block'], target)
    skill_done = opt['skill_done'] | (
        opt['valid'] & jnp.where(opt['pick'], picked, placed))
    tower = (state['k'] == state['n']) & (opening > scene.OPEN_WIDTH)
    limit = jnp.where(opt['valid'], SKILL_STEPS, INVALID_STEPS)
    end = ((skill_done & ~tower) | ((steps >= limit) & ~skill_done) |
           new_obs['is_last'])
    running = ran & ~end
    opt = dict(opt, steps=steps, reward=reward, skill_done=skill_done,
               running=running)
    return state, opt, obs, running.any()

  def _select(self, mask, new, old):
    """`new` for the worlds in `mask`, `old` for the rest."""
    nd, od = new['data'], old['data']
    m = mask[:, None]

    def freeze():
      return self.base.sim.forward(nd.replace(
          qpos=jnp.where(m, nd.qpos, od.qpos),
          qvel=jnp.where(m, nd.qvel, od.qvel),
          ctrl=jnp.where(m, nd.ctrl, od.ctrl)))

    d = jax.lax.cond(mask.all(), lambda: nd, freeze)
    rest = {k: _pick(mask, new[k], old[k])
            for k in new if k not in ('data', 'key')}
    return dict(rest, data=d, key=new['key'])


class StackOptions(stack.Stack):

  @property
  def env(self):
    if self._env is None:
      self._env = BatchStackOptions(self.batch, self.size, seed=self.seed,
                                    **self.options)
    return self._env

  @property
  def obs_space(self):
    return obs_space(self.size)

  @property
  def act_space(self):
    return act_space()
```

  In `environments/__init__.py` add
  `'stack_options': 'environments.stack_options:StackOptions',` to
  `REGISTRY` (it shares the `stack` config section through `OPTIONS`).
  Add a short "Stack options" section to `environments/README.md` giving
  the option numbering, validity, `log/duration` and the frozen-world rule.

  Implementation notes for the engineer:
  - `_advance_impl` already computes the episode's reward, prefix,
    success and time limit; reuse it, do not reimplement them.
  - The `still` action for a world at its first tick is replaced by
    `HOLD` inside `_advance_impl`, so first worlds behave as in
    `BatchStack`.
  - If `jax.lax.cond` over the Warp data fails to compile, call `freeze()`
    unconditionally (one extra forward pass per tick).

- [ ] **Step 4: Run the tests** (GPU; nothing else on it)

Run: `$PY -m pytest tests/test_stack_options.py tests/test_stack_env.py -q`
Expected: all pass. `test_pick_lifts_the_block` uses the trained students
and allows one of four worlds to fail.

- [ ] **Step 5: Commit**

```bash
git add environments/stack_options.py environments/__init__.py environments/README.md tests/test_stack_options.py
git commit -m "Add the option environment for block stacking"
```

### Task 4: Scripted towers through the options

The spec's check that the skills chain: options chosen from the simulator
state build towers. Its mean option duration sets the option pilot's
`run.steps`.

**Files:**
- Create: `skills/tower.py`
- Test: `tests/test_tower.py`

**Interfaces:**
- Consumes: `BatchStackOptions`, `privileged()['option_mask']`,
  `privileged()['blocks']`, `privileged()['held']`.
- Produces: `choose(blocks, held, mask, goal, n, k) -> int` (one world);
  `outputs/eval/stack_options/scripted_n<N>.json` with `success`,
  `option_steps` (mean control steps of valid options), `options` and
  `control_steps` per episode.

- [ ] **Step 1: Write the failing test** `tests/test_tower.py` (pure
  numpy, no GPU):

```python
import numpy as np

from environments.stack_scene import HALF
from skills.tower import choose

TABLE = 8


def blocks(**xyz):
  out = np.zeros((4, 11))
  for c, k in enumerate('rgby'):
    if k in xyz:
      out[c, 0] = 1
      out[c, 1:4] = xyz[k]
  return out


def mask(b, held):
  present = b[:, 0] > 0
  clear = present.copy()
  for c in range(4):
    for o in range(4):
      if (o != c and present[o] and o != held and
          np.linalg.norm(b[o, 1:3] - b[c, 1:3]) < 0.03 and
          b[o, 3] > b[c, 3] + HALF):
        clear[c] = False
  clear[held] = False if held >= 0 else clear[held]
  holding = held >= 0
  return np.concatenate([clear & ~holding, clear & holding, [holding]])


GOAL = np.array([0, 1, 2, 4])  # red, green, blue bottom to top.


def test_picks_the_next_goal_block():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.6, 0, HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 1) == 1


def test_places_the_next_block_on_the_tower():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, 0.12), b=(0.6, 0, HALF))
  assert choose(b, 1, mask(b, 1), GOAL, 3, 1) == 4 + 0


def test_puts_a_wrong_block_on_the_table():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.6, 0, 0.12))
  assert choose(b, 2, mask(b, 2), GOAL, 3, 1) == TABLE


def test_clears_a_block_off_the_tower():
  b = blocks(r=(0.4, 0, HALF), g=(0.5, 0, HALF), b=(0.4, 0, 3 * HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 1) == 2


def test_moves_a_misplaced_base_to_the_table():
  b = blocks(r=(0.5, 0, 3 * HALF), g=(0.5, 0, HALF), b=(0.6, 0, HALF))
  assert choose(b, -1, mask(b, -1), GOAL, 3, 0) == 0
  assert choose(b, 0, mask(b, 0), GOAL, 3, 0) == TABLE


def test_waits_once_the_tower_is_built():
  b = blocks(r=(0.4, 0, HALF), g=(0.4, 0, 3 * HALF), b=(0.4, 0, 5 * HALF))
  option = choose(b, -1, mask(b, -1), GOAL, 3, 3)
  assert not mask(b, -1)[option]
```

- [ ] **Step 2: Run it to see it fail**

Run: `$PY -m pytest tests/test_tower.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement** `skills/tower.py`:

```python
"""Build towers through the option environment, choosing options from the
simulator state: the check that the trained skills chain into stacks.

  python skills/tower.py --n 3
  python skills/tower.py --n 2 --episodes 128

Plays one episode in each of --episodes worlds and writes
outputs/eval/stack_options/scripted_n<n>.json.
"""

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse  # noqa: E402
import json  # noqa: E402

import numpy as np  # noqa: E402

COLOURS = 4
TABLE = 2 * COLOURS
NEAR_XY = 0.03


def top_of(blocks, c):
  """The highest present block in c's stack, at or above c."""
  pos = blocks[:, 1:4]
  stack = ((blocks[:, 0] > 0) &
           (np.linalg.norm(pos[:, :2] - pos[c, :2], axis=-1) < NEAR_XY) &
           (pos[:, 2] >= pos[c, 2]))
  return int(np.argmax(np.where(stack, pos[:, 2], -1.0)))


def choose(blocks, held, mask, goal, n, k):
  """The next option for one world: clear the tower top and the next goal
  block, then pick the next block and put it on the tower."""
  if k >= n:
    return int(np.flatnonzero(~mask)[0])  # Hold still: any invalid option.
  want = int(goal[k])
  if held >= 0:
    if held == want:
      return TABLE if k == 0 else COLOURS + int(goal[k - 1])
    return TABLE
  for c in ([int(goal[k - 1])] if k > 0 else []) + [want]:
    if not mask[c]:
      return top_of(blocks, c)
  return want


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--n', type=int, default=3, help='blocks per tower')
  parser.add_argument('--episodes', type=int, default=64)
  parser.add_argument('--seed', type=int, default=1)
  parser.add_argument('--max-options', type=int, default=60,
                      help='decisions before an episode counts as failed')
  args = parser.parse_args()
  from environments.stack_options import BatchStackOptions
  env = BatchStackOptions(args.episodes, n=args.n, seed=args.seed)
  b = args.episodes
  obs = env.step({'reset': np.ones(b, bool), 'action': np.zeros(b, np.int32)})
  ended = np.zeros(b, bool)
  success = np.zeros(b, bool)
  options = np.zeros(b, int)
  steps = np.zeros(b, int)
  durations = []
  for _ in range(args.max_options):
    priv = env.privileged()
    action = np.array([choose(
        priv['blocks'][i], priv['held'][i], priv['option_mask'][i],
        obs['goal'][i], obs['log/n'][i], obs['log/prefix'][i])
        for i in range(b)], np.int32)
    valid = priv['option_mask'][np.arange(b), action]
    obs = env.step({'reset': np.zeros(b, bool), 'action': action})
    live = ~ended
    options += live
    steps += np.where(live, obs['log/duration'], 0)
    durations += list(obs['log/duration'][live & valid])
    success |= live & obs['is_terminal']
    ended |= obs['is_last']
    if ended.all():
      break
  result = dict(
      n=args.n, episodes=b, seed=args.seed,
      success=float(success.mean()),
      option_steps=float(np.mean(durations)),
      options=float(options.mean()), control_steps=float(steps.mean()))
  path = REPO / 'outputs' / 'eval' / 'stack_options' / f'scripted_n{args.n}.json'
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2))
  print(json.dumps(result))


if __name__ == '__main__':
  main()
```

  The episode itself restarts a world once it ends; `ended` keeps those
  worlds out of the counts. A world that keeps going after its episode
  restarts is harmless (its options are not counted).

- [ ] **Step 4: Run the unit test, then the towers** (GPU)

Run: `$PY -m pytest tests/test_tower.py -q` (pass), then
`$PY skills/tower.py --n 3` and `$PY skills/tower.py --n 2 --episodes 128`.
Expected: N = 3 success ≥ 0.7 (about 0.95 per skill over five or more
options). Below that, look at which option fails (log the failing option
per world) before going on: Dreamer cannot learn more than the skills
allow.

- [ ] **Step 5: Commit**

```bash
git add skills/tower.py tests/test_tower.py outputs/eval/stack_options
git commit -m "Build towers through the options from the simulator state"
```

### Task 5: Option Dreamer pilot

**Files:**
- Modify: `AGENTS.md` (the option task beside the `custom_stack` notes),
  spec "Build order" stage 4 (the measured mean duration and budget).

- [ ] **Step 1: Smoke run** (GPU, about 5 min):

```bash
MALLOC_MMAP_THRESHOLD_=1048576 $PY train.py --task custom_stack_options \
  --configs debug --env.stack.n 2 --run.envs 4 --run.steps 400 \
  --jax.prealloc False --logdir $SCRATCH/options_smoke
```

  Expected: it runs to 400 steps, `metrics.jsonl` has `episode/score`, and
  the console shows the `log/duration` per-episode stats.

- [ ] **Step 2: Budget.** The raw pilot `stack_raw_n2` ran 2,571,120
  control steps. `run.steps` for the option pilot is that divided by the
  N = 2 `option_steps` from Task 4, rounded up to a thousand. Record the
  number in the spec.

- [ ] **Step 3: Launch the pilot** (local GPU, in the background, with
  the GPU to itself), the raw pilot's flags but the task and budget:

```bash
MALLOC_MMAP_THRESHOLD_=1048576 nohup $PY train.py \
  --task custom_stack_options --env.stack.n 2 --run.envs 16 \
  --replay.size 2.5e5 --jax.prealloc False --run.steps <budget> \
  --logdir outputs/train/stack_options_n2 \
  > outputs/logs/stack_options_n2.log 2>&1 &
```

- [ ] **Step 4: Document and commit** the AGENTS.md note and the spec
  numbers (not the running run's outputs):

```bash
git add AGENTS.md docs/superpowers/specs
git commit -m "Note the option task and the option pilot's budget"
```

  The pilot's results, the comparison against `stack_raw_n2` at equal
  `log/sim_steps`, and the main runs over N in {2, 3, 4} are read off once
  it finishes; stage 5 starts from them.
