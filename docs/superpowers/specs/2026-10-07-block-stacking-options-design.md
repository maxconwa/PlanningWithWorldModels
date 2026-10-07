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
  with its licence. The scene (table, blocks) is our own MJCF.
- **Physics** at 500 Hz; **control** at 20 Hz (25 physics steps per control
  step).
- **Arm controller.** Differential inverse kinematics from an end-effector
  target to joint position targets for the Panda's position actuators, with
  the gripper pointing down and only yaw free.

## Observations

Full state, no images, the same for both agents:

| Key | Contents |
|---|---|
| `arm` | joint positions and velocities, end-effector position and yaw, gripper opening |
| `blocks` | per colour slot: present flag, position, orientation (quaternion), linear velocity; zeros when absent |
| `goal` | goal order as 4 colour ids bottom to top, padded with a "none" id |
| `progress` | tower prefix k and N |

The option environment adds two more:

| Key | Contents |
|---|---|
| `option_mask` | which of the 9 options are valid now (for planners and logging) |
| `duration` | control steps the last option took |

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

Two goal-conditioned skill policies, trained with PPO (Stable-Baselines3, on
the CPU, many parallel MuJoCo environments) on the raw action space:

- **Pick.** Input: state observation and the target colour. Success: block
  grasped and lifted 5 cm clear of anything below it.
- **Place.** Input: state observation and the target (a colour, or the
  table). Success: held block released and resting on the target, gripper
  open and raised 5 cm.

Both output the 5 raw action dimensions plus a **terminate** logit. Choosing
to terminate ends the skill: it earns a bonus if the success condition holds
and a penalty otherwise. Skill training uses its own dense shaped reward
(reach, grasp, lift or align, release). This reward is for the skills only;
the stacking agents never see it.

Start states for skill training are drawn from the same distribution as
stacking episodes. Place starts with a block already in the gripper, set
directly in the simulator state. **Gate:** each skill must succeed in at
least 90% of 500 test episodes before the option agent is trained.

## Agents and training

- **DreamerV3** (`third_party/dreamerv3` at `e3f0224`, unchanged), launched
  through `train.py` with `--task custom_stack` (raw) and
  `--task custom_stack_options` (options). Same config for both: `size12m`,
  `train_ratio` 32, MLP encoder and decoder on the state keys, `contdisc`.
  The only difference is the action space (bounded normal vs categorical
  policy). The option agent's discount applies per decision, not per elapsed
  control step; this is part of what "option space" means here.
- **Budget.** Equal low-level control steps. Both environments report the
  running total of control steps (`log/sim_steps`). DreamerV3's
  `run.steps` counts agent decisions, so the option run's `run.steps` is set
  from the pilot's mean option duration, and both runs are compared at the
  same `sim_steps` on their training curves.
- **Skill accounting.** Results are reported twice: with skills treated as
  given, and with the control steps spent training the skills added to the
  option agent's total.
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
  by the recorded low-level actions). Reported as:
  - Block pose error at matched sim time (after 1, 2 and 3 options' worth of
    control steps).
  - Error per decision (after k decisions on each model's own time scale).
  - Return-prediction error: predicted vs actual episode return from the
    start state.
- **Tree figures.** `planning/search_tree.py` draws the option agent's trees
  as it does for Tetris placements.

## Code layout

- `environments/stack.py`: the MuJoCo scene, controller, task, reward and
  success (`custom_stack`, raw actions).
- `environments/stack_options.py`: the option environment
  (`custom_stack_options`). It wraps `stack.py` and loads the trained skills.
- `environments/assets/panda/`, `environments/assets/stack.xml`: the robot
  and scene.
- `skills/train.py`, `skills/envs.py`: PPO training of pick and place on
  `stack.py`, and the skill checkpoints' format.
- `planning/search.py`: iCEM and sampled MuZero.
- `planning/eval_model.py`: custom environments, continuous actions, success
  by N.
- `planning/model_error.py`: model accuracy.
- New dependencies in the `dreamerv3` conda env: `mujoco`,
  `stable-baselines3` and CPU-only `torch`.

## Build order

Each step ends with a check before the next starts.

1. **Scene and controller.** Panda, table and blocks load; the IK controller
   tracks end-effector targets. Check: drive the arm by hand through
   `environments/play.py`.
2. **Raw stacking env.** Observations, reward, prefix, success, start states.
   Check: unit tests on hand-built states (correct and wrong towers, held
   block, toppled tower), and a `--configs debug` DreamerV3 run of 400 steps.
3. **Skills.** PPO pick and place with learned termination. Check: the 90%
   gate on 500 test episodes each.
4. **Option env.** Masks, invalid-option cost, duration, summed reward,
   time-limit cut-off. Check: unit tests, plus a scripted sequence of valid
   options that builds a correct N = 3 tower.
5. **Pilot.** N = 2 only, both agents, one seed, local. Measures mean option
   duration and learning speed, to set the step budget.
6. **Main runs.** N in {2, 3, 4}, both agents, one seed, at the budget from
   the pilot.
7. **Planners.** iCEM and sampled MuZero. Check: on the option side, sampled
   MuZero with K = 9 matches PUCT within noise, and both planners beat a
   random-action baseline on the trained models.
8. **Evaluation.** The 2 x 3 table (plus the extra PUCT and Gumbel rows), the
   skill-accounting variant, and the model-accuracy measures.

## Out of scope

Images, real hardware, more seeds or Delta scaling (decided after the pilot),
learned or discovered skills beyond pick and place, and changes to the pinned
DreamerV3.
