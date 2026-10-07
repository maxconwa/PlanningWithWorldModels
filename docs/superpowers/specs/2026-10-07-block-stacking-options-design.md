# Block stacking: planning in raw vs option action spaces

## Goal

Find out whether planning in a learned world model works better when the
agent acts through options than through raw arm actions, on robot block
stacking. The Tetris runs suggested it does (placement options with PUCT beat
the options policy; search did nothing for joystick actions), but those runs
differed in model size, training steps and piece order as well as in action
space. This experiment holds everything but the action space equal.

The result is one table, {raw, options} x {policy, iCEM, sampled MuZero},
supporting or refuting three claims:

1. **Planning gain.** Planner minus policy success is larger with options.
2. **Model accuracy.** The option world model predicts the future more
   accurately than the raw one.
3. **End to end.** Options reach higher stacking success than raw actions at
   the same number of low-level simulator steps.

Everything that acts or predicts works from camera images: the world models,
the end-to-end raw policy, and the skills behind the options all see a fixed
workspace camera and a wrist camera, plus proprioception and the goal.

Sim only. Realism matters less than a fair, repeatable comparison.

## Task

A Franka Panda in MuJoCo stacks N coloured blocks into a tower in a goal
order, bottom to top.

- **Blocks.** Four colours (red, green, blue, yellow), 4 cm cubes. Each
  episode samples N uniformly from {2, 3, 4} and a goal order of N distinct
  colours. Only those N blocks are in the scene.
- **Start states.** Arm at its home pose. Blocks at random non-overlapping
  positions and yaws in a 30 x 30 cm workspace. In 30% of episodes, 2 or 3
  of the blocks start as a random stack (which may or may not match a prefix
  of the goal), so the agent sometimes has to unstack.
- **Time limit.** 20 s of sim time for N = 2, 30 s for N = 3, 40 s for N = 4,
  the same for both agents. An option that would cross the limit is cut off
  at it.
- **Tower prefix.** The number k of goal blocks correctly in place: the first
  goal block rests on the table, each next one rests on the one below it
  (centres within 1.5 cm horizontally), and none of them is held by the
  gripper.
- **Reward (both agents).** +1 each time k increases, -1 each time it
  decreases, and +1 when the episode succeeds. The raw agent gets it per
  control step; the option agent gets the sum over the option.
- **Success.** k = N, still true 1 s of sim time after it is first reached,
  with the gripper open. Success ends the episode.

## Simulator

- **MuJoCo** (`mujoco` Python bindings), the Panda and its gripper from
  MuJoCo Menagerie (Apache 2.0), copied into `environments/assets/panda/`
  with its licence. The scene (floor as the table, blocks, cameras) is our
  own MJCF.
- **Model file.** Menagerie's MJX variant of the Panda.
- **Everything on the GPU.** Physics and rendering run in MJX with its Warp
  backend (`mujoco-mjx[warp]`): MuJoCo Warp for the physics and MuJoCo Warp's
  batch renderer for the cameras, as JAX functions over a batch of worlds.
  Every stage (skill training, both Dreamer agents, planning and
  evaluation) uses this one simulator and renderer, so a skill never sees
  images from a different renderer than the one it was trained on. CPU
  MuJoCo is used only in tests, as a reference for the model.
- **Not Madrona.** Madrona-MJX is deprecated and needs JAX < 0.6 and CUDA
  <= 12.5; the local RTX 5070 Ti (Blackwell) needs CUDA >= 12.8 and runs JAX
  0.10.2. Its README points to MuJoCo Warp's renderer, used here.
- **Physics** at 200 Hz with Menagerie's MJX solver settings (timestep
  0.005, implicitfast, 5 iterations); **control** at 20 Hz (10 physics steps
  per control step).
- **Cameras.** A fixed workspace camera looking down at the table at about
  45 degrees, and a wrist camera on the gripper looking along it. Both
  render 64 x 64 RGB once per control step.
- **Arm controller.** Differential inverse kinematics from an end-effector
  target to joint position targets for the Panda's position actuators, with
  the gripper pointing down and only yaw free. The Jacobian comes from
  differentiating MJX's (JAX backend) forward kinematics, so the controller
  is a JAX function too.
- **Batched environments.** One environment object holds a batch of worlds
  and steps them together on the GPU; worlds reset independently when their
  episode ends. A single-world wrapper serves manual play and tools that
  expect one environment.

## Observations

The same for both agents and for the skill policies:

| Key | Contents |
|---|---|
| `workspace` | 64 x 64 RGB from the workspace camera |
| `wrist` | 64 x 64 RGB from the wrist camera |
| `arm` | joint positions and velocities, end-effector position and yaw, gripper opening |
| `goal` | goal order as 4 colour ids bottom to top, padded with a "none" id |

Block poses come only from the images. Scalar diagnostics go along as `log/`
keys, which DreamerV3 keeps out of the model (`main.py` drops them from the
agent's observation space) and logs per episode. Its training loop accepts
only scalar `log/` keys:

| Key | Contents |
|---|---|
| `log/prefix` | tower prefix k |
| `log/n` | number of blocks N |
| `log/success` | 1 on the step the episode succeeds |
| `log/sim_steps` | running total of control steps in this environment |

The option environment adds `log/duration`, the control steps the last option
took, so both world models get exactly the same inputs.

The simulator's full state is not an observation. Each environment returns it
from a `privileged()` method, which the skill teachers, the planners (for the
option mask), the pose probes and evaluation call directly:

| Key | Contents |
|---|---|
| `blocks` | per colour slot: present flag, position, orientation (quaternion), linear velocity; zeros when absent |
| `held` | colour of the held block, or -1 |
| `option_mask` | which of the 9 options are valid now (option environment only) |

## Raw action space

Continuous, 5 dimensions in [-1, 1]: end-effector position delta (dx, dy, dz,
up to 2 cm per control step), yaw delta (up to 0.2 rad), and gripper (> 0
closes). One action per control step.

## Option action space

Discrete, 9 options:

- `pick(c)` for each colour c: pick up block c.
- `place_on(c)` for each colour c: put the held block on block c.
- `place_on(table)`: put the held block on a free spot on the table.

An option is valid only when it makes sense: `pick` with an empty gripper and
block c present with nothing on top of it; `place_on(c)` while holding a
block other than c, with c present and clear; `place_on(table)` while holding
a block. An invalid option does nothing for 10 control steps, so it is never
free.

Each option runs a learned skill (below) on the raw action space until the
skill signals that it is done, or for at most 150 control steps.

## Skills

Two goal-conditioned skills on the raw action space:

- **Pick(c).** Success: block c grasped and lifted 5 cm clear of anything
  below it.
- **Place(target).** Target is a colour or the table. Success: held block
  released and resting on the target, gripper open and raised 5 cm.

Each skill outputs the 5 raw action dimensions plus a **terminate** logit.
Choosing to terminate ends the skill: it earns a bonus if the success
condition holds and a penalty otherwise.

Each skill is trained in two stages:

1. **State teacher.** PPO from privileged state (arm, block poses, target),
   with Brax's PPO on the batched simulator: thousands of worlds on the GPU,
   no rendering. It uses its own dense shaped reward (reach, grasp, lift or
   align, release), which the stacking agents never see.
2. **Image student.** The deployed skill: a CNN policy on `workspace`,
   `wrist`, `arm` and the target, in JAX/Flax. It is trained with DAgger on
   the batched simulator with rendering, on the GPU: the student acts, the
   teacher labels each visited state with its action and terminate decision,
   and the student is trained on the growing dataset.

Start states for skill training are drawn from the same distribution as
stacking episodes. Place starts with a block already in the gripper, set
directly in the simulator state.

**Gates,** each on 500 test episodes on the simulator the option environment
uses:

- the teacher succeeds in at least 90% before distillation;
- the student succeeds in at least 90% before the option agent is trained.

## Agents and training

- **Stages.** Training runs in stages, each with the GPU to itself:
  (1) environment and tests, (2) skills, (3) the raw-action Dreamer agent,
  (4) the option Dreamer agent, (5) planning and evaluation. Within a
  Dreamer stage the agent and the batched environment share the GPU in one
  process, with JAX preallocation off.
- **Batched driver.** DreamerV3's `embodied.Driver` steps environments one at
  a time. `train.py` swaps in a driver with the same interface that steps
  one batched environment (all `run.envs` worlds at once), as it already
  swaps in `make_env`. DreamerV3 itself is unchanged.
- **DreamerV3** (`third_party/dreamerv3` at `e3f0224`, unchanged), launched
  through `train.py` with `--task custom_stack` (raw) and
  `--task custom_stack_options` (options). Same config for both: `size12m`,
  `train_ratio` 32, `contdisc`. The encoder and decoder take both 64 x 64
  cameras (CNN) and `arm` and `goal` (MLP).
  The only difference is the action space (bounded normal vs categorical
  policy). The option agent's discount applies per decision, not per elapsed
  control step; this is part of what "option space" means here.
- **Budget.** Equal low-level control steps. Both environments report the
  running total of control steps (`log/sim_steps`). DreamerV3's
  `run.steps` counts agent decisions, so the option run's `run.steps` is set
  from the pilot's mean option duration, and both runs are compared at the
  same `sim_steps` on their training curves.
- **Skill accounting.** Results are reported twice: with skills treated as
  given, and with the control steps spent training the skills (teacher PPO
  steps in MJX plus student DAgger rollout steps) added to the option agent's
  total.
- **Seeds and compute.** One seed per condition, on the local GPU. Scaling to
  more seeds on Delta is a separate decision after the pilot.

## Planners

Added to `planning/search.py`, each with module-level defaults like `PUCT`.
Each one works on both action spaces:

- **iCEM.** Plans an action sequence in the world model's latent space over a
  horizon H. Raw: it samples continuous sequences with coloured noise, keeps
  the elites, refits a Gaussian per step, and warm-starts from the policy and
  the shifted previous plan. Options: it refits a categorical distribution
  per step instead. Sequences are scored by predicted reward plus the critic
  at the end. It executes the first action and replans every step.
- **Sampled MuZero.** At each node it samples K actions from the policy
  (without replacement for options, so K up to 9 covers every option) and
  runs PUCT over them with the policy as the prior. It plays the most
  visited root action. Raw: the K sampled actions are stored in the node so
  `recurrent_fn` can look them up.

On the option side, the existing PUCT and Gumbel planners also run, as extra
rows.

## Evaluation

- **Success and reward.** `planning/eval_model.py` is extended to custom
  environments and continuous actions. It reports success rate and mean
  stacking reward per N. Each agent and planner plays 100 episodes per N in
  {2, 3, 4}, from the same 300 fixed start states and goals (fixed seeds).
- **Model accuracy** (`planning/model_error.py`, new). Held-out episodes are
  played with the option agent's policy, and the low-level actions are
  recorded during each option. Both models are then rolled open-loop through
  the same executed behaviour (the option model by its options, the raw model
  by the recorded low-level actions).

  Block poses are read out of each model's latent states by a **pose probe**:
  a small MLP per model, fitted on posterior latents of separate held-out
  real episodes to the true block poses from `privileged()`, then frozen. Reported as:
  - Block pose error at matched sim time (after 1, 2 and 3 options' worth of
    control steps): probe on the open-loop prediction vs the simulator.
  - Error per decision (after k decisions on each model's own time scale).
  - Image error of the decoded `workspace` and `wrist` frames vs the real
    ones, at the same points.
  - Return-prediction error: predicted vs actual episode return from the
    start state.
  - The probe's own error on posterior latents, as the floor the open-loop
    errors are read against.
- **Tree figures.** `planning/search_tree.py` draws the option agent's trees
  as it does for Tetris placements.

## Code layout

- `environments/assets/panda/`, `environments/assets/stack.xml`: the robot
  and scene.
- `environments/stack_scene.py`: building the model (with the wrist camera).
- `environments/stack_sim.py`: the batched GPU simulator: physics substeps
  and rendering.
- `environments/stack_control.py`: the end-effector controller (JAX).
- `environments/stack_task.py`: start states, tower prefix, held block,
  reward and success (JAX).
- `environments/stack.py`: the batched environment `BatchStack` and the
  single-world `Stack` (`custom_stack`, raw actions).
- `environments/batch_driver.py`: the batched driver `train.py` swaps in.
- `environments/stack_options.py`: the option environment
  (`custom_stack_options`). It wraps `BatchStack` and runs the trained
  skills.
- `skills/env.py`: the pick and place tasks on `BatchStack`'s simulator.
- `skills/teacher.py`: Brax PPO training of the state teachers.
- `skills/student.py`: the image policy network and DAgger distillation.
- `skills/policy.py`: loading a trained student for the option environment.
- `planning/search.py`: iCEM and sampled MuZero.
- `planning/eval_model.py`: custom environments, continuous actions, success
  by N.
- `planning/model_error.py`: model accuracy.
- New dependencies in the `dreamerv3` conda env: `mujoco`, `mujoco-mjx[warp]`
  (which brings `warp-lang`) and `brax`, installed with JAX pinned at 0.10.2
  (no torch).

## Build order

Five stages, each with the GPU to itself. Each step ends with a check before
the next starts.

1. **Environment and tests.** Install the Warp backend and check it renders on
   this GPU; the scene; the batched simulator; task logic; the controller;
   `BatchStack` and `Stack`; manual play; the batched driver and a short
   DreamerV3 run on `custom_stack`. Checks: unit tests, grasp-and-lift, and
   the Dreamer run.
2. **Skills.** State teachers by Brax PPO, then image students by DAgger,
   both with learned termination, all on the GPU. Checks: the teacher gate,
   then the student gate.
3. **Raw-action agent.** DreamerV3 on `custom_stack`: an N = 2 pilot to
   measure learning speed, then the main run over N in {2, 3, 4}.
4. **Option agent.** The option environment (masks, invalid-option cost,
   duration, summed reward, time-limit cut-off; checked by tests and a
   scripted N = 3 tower), then DreamerV3 on `custom_stack_options`: the N = 2
   pilot sets the budget from the mean option duration, then the main run at
   the same control-step budget as stage 3.
5. **Planning and evaluation.** iCEM and sampled MuZero (sampled MuZero with
   K = 9 matches PUCT within noise on the option side; both beat a random
   baseline), the 2 x 3 table plus the extra PUCT and Gumbel rows, the
   skill-accounting variant, the pose probes, and the model-accuracy
   measures.

## Out of scope

Real hardware, more seeds or Delta scaling (decided after the pilot),
learned or discovered skills beyond pick and place, and changes to the pinned
DreamerV3.
