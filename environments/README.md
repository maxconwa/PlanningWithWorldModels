# Custom environments

Training environments for world models, run by DreamerV3 through `train.py`
at the repository root:

    python train.py --task custom_catch --configs debug --logdir /tmp/catch

`--task custom_<name>` makes the environment registered as `<name>` in
`__init__.py`; every other `--task` goes to DreamerV3's own environments, and
all other flags are DreamerV3's.

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
