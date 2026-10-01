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

## Playing by hand

    python environments/play.py tetris
    python environments/play.py tetris --option gray=False --option sticky=False

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
