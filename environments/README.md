# Custom environments

Training environments for world models, run by DreamerV3 through `train.py`
at the repository root:

    python train.py --task custom_catch --configs debug --logdir /tmp/catch

`--task custom_<name>` makes the environment registered as `<name>` in
`__init__.py`; every other `--task` goes to DreamerV3's own environments, and
all other flags are DreamerV3's.

## Environments

- `catch`: a paddle catches falling balls on an 8x8 grid. Small enough to
  train in minutes, as a check that a world model learns at all.
- `tetris`: Atari Tetris with placement actions. Actions 0-9 put the falling
  piece's leftmost cell in that column and drop it; action 10 rotates it
  once. Pieces are random (`random_pieces=True`); the game on its own deals
  a fixed 16-piece cycle, which an agent can memorise. All other options are
  DreamerV3's Atari ones, set with `--env.atari.*`. Trained by
  `slurm/tetris_placement.slurm`.
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
- `stack_options`: the same task through 9 options, each running a trained
  image skill from `skills/` (`outputs/train/skill_{pick,place}_student`).
  Options 0-3 pick a colour, 4-7 place the held block on a colour, 8 on the
  table. An option is valid when it makes sense (pick with an empty
  gripper and a clear block; place while holding, onto a clear block other
  than the held one, or the table); `privileged()['option_mask']` says
  which are. A valid option runs until its skill succeeds or for 150
  control steps, an invalid one holds still for 10, and a place that
  completes the tower holds still until success is decided; all end with
  the episode. The reward is summed over the option and `log/duration`
  counts its control steps. Worlds whose option has ended are frozen until
  every world's has. Takes the `stack` options (`--env.stack.n`).

## Playing by hand

    python environments/play.py tetris
    python environments/play.py tetris --option gray=False --option sticky=False
    python environments/play.py stack --scale 6

Opens a window with the emulator screen (for Atari environments) and the
observation the agent gets, using the options training uses unless
`--option key=value` overrides them. The keys for each environment are shown
in the window and listed in `KEYS` in `play.py`.

## Adding an environment

1. Write a class in a new file here that subclasses `embodied.Env` (see
   `catch.py`). Its constructor takes the task name first, then keyword
   options. It needs:
   - `obs_space`: a dict of `elements.Space`, with `image` (uint8, HxWx3) or
     vector inputs, plus `reward`, `is_first`, `is_last` and `is_terminal`.
   - `act_space`: a dict with `reset` (bool) and `action`.
   - `step(action)`: on `action['reset']`, or after an episode ends, start a
     new episode and return its first observation with `is_first=True`.
2. Add it to `REGISTRY` in `__init__.py` as `'name': 'environments.file:Class'`.
3. Add its keys to `KEYS` and `HELP` in `play.py`, then try it by hand.
4. Options come from `config.env.custom`, or from the config section named by
   a class attribute `OPTIONS`; `tetris` sets `OPTIONS = 'atari'` to share
   DreamerV3's Atari options.
