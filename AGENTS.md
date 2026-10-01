# Agent guide

## Goal

This project looks for algorithms and models that support planning with world
models: which world models can be planned in, and which planners get the most
out of them. The loop is to train world models (on NCSA Delta, with Slurm),
then plan inside them and measure how the plans play on the real environment
against the model's own policy.

Two families of world model:

- **DreamerV3** (`third_party/dreamerv3`): an RSSM agent trained online by
  playing the environment. Every current run in `outputs/train/` is DreamerV3
  on Atari (Breakout, Tetris), and all code in `planning/` targets it.
- **Open Dreamer** (`third_party/open-dreamer`): a JAX/Flax implementation of
  Dreamer 4, with a causal video tokenizer and an action-conditioned latent
  dynamics model, trained offline on gameplay video. It has its own
  dependencies (`pyproject.toml`, `uv.lock`, Python 3.11) and training
  scripts (`scripts/train_tokenizer.py`, `scripts/train_dynamics.py`). None of
  this repository's code uses it yet.

A planner result counts when it comes from full episodes on the real
environment (`planning/eval_model.py`), compared with acting by the policy
under the same emulator settings and number of episodes.

## Layout

- `third_party/`: the two world models as git submodules. `dreamerv3` is
  pinned at `e3f0224`, the commit the checkpoints in `outputs/train/` were
  trained with. Do not edit a submodule or move its pin without asking:
  checkpoints may stop loading. Extend them from this repository instead, as
  `train.py` does.
- `environments/`: custom training environments (`embodied.Env` subclasses),
  listed in `REGISTRY` in `environments/__init__.py` and trained with
  `python train.py --task custom_<name>`. `train.py` wraps DreamerV3's
  `main.py`, adds the `custom` suite, and passes every other task through.
- `planning/`: planning in trained world models, run as
  `python planning/<script>.py` from the repository root.
  - `world_model.py` loads a run (`build`) and drives the real emulator to a
    state to plan from (`warmup`). `logdir_arg` finds a bare run name in
    `outputs/train/`; `eval_path` puts a bare `--out` file name in
    `outputs/eval/<run>/`. Every planning script uses both.
  - `search.py` holds the planners (`make_search`: `gumbel`, `puct`, `uct`).
    New planners go here, with their default settings as a module-level dict
    like `PUCT` and `UCT`.
  - `eval_model.py`, `search_tree.py`, `dream_atari.py` are the scripts.
  - The planners call DreamerV3's model directly (`dyn.imagine` and the
    `rew`, `con`, `val`, `pol` heads). A second world model should be loaded
    by its own module that exposes the same operations, rather than by
    branches inside the planners.
- `slurm/`: Delta job scripts, `activate.sh`, and the local helpers
  `ssh_delta.sh` and `sync_from_delta.sh`.
- `outputs/`: `train/<run>/` holds training runs (DreamerV3 logdirs with the
  latest checkpoint only), `eval/<run>/` evaluation and planning results for
  that run, and `logs/` Slurm logs.
- `jsonl_to_tensorboard.py`: turns a run's `metrics.jsonl` into TensorBoard
  logs.

## Adding things

- **Training job**: copy an existing `slurm/*.slurm` and keep its `#SBATCH`
  header, `source .../slurm/activate.sh` and `cd $REPO/third_party/dreamerv3`.
  Use `python $REPO/train.py` in place of `python dreamerv3/main.py` for a
  custom environment. Write to `--logdir $NVME/logdir/<run>`.
- **Run names**: lowercase with underscores, task first, then what sets the
  run apart (`tetris_400m`, `breakout_69k`). The name on Delta and in
  `outputs/train/` is the same, so `sync_from_delta.sh <run>` finds it.
- **Environment**: a file in `environments/` plus a `REGISTRY` entry; see
  `environments/README.md` for the interface.
- **Planning script**: in `planning/`, starting with a docstring of example
  command lines, taking `--logdir` with `type=logdir_arg` and writing through
  `eval_path`.

## Storage

- Everything under `outputs/` is stored in Git LFS (`.gitattributes`), as are
  `*.pkl` files anywhere.
- GitHub rejects LFS files over 2 GiB. Checkpoints that large go in
  `.gitignore`. The tetris `agent.pkl` files are, so those runs do not load
  from a fresh clone.
- Never commit replay buffers (`replay/`, `replay_old/`).

## Python environments

- Local: the `dreamerv3` conda env (`~/miniconda3/envs/dreamerv3`) with an
  RTX 5070 Ti (16 GB). The GPU needs a newer JAX (0.10.2 here) than the
  pinned DreamerV3 targets (0.4.33, which cannot run on it). `jax_compat.py`
  bridges the one difference, keyword-only `jax.jit` options; `train.py` and
  `planning/world_model.py` import it, and new entry points that build a
  DreamerV3 agent must too. Without it building an agent fails with
  `TypeError: jit() takes from 0 to 1 positional arguments`. The env lacks
  `pygame`, which `dream_atari.py` needs.
- JAX takes most of the GPU's memory when a process starts, so only one
  training or agent-loading process can use the local GPU at a time; a
  second one fails with `RESOURCE_EXHAUSTED`. Check `nvidia-smi` before
  starting one, and do not start one while a training run is going.
- Local training writes to `outputs/train/<run>/` and logs to
  `outputs/logs/<run>.log`; its replay buffer (`replay/`) is git-ignored.
- Delta: `slurm/activate.sh` loads the `/projects/biny/mconway/dreamer-env`
  conda env and sets `PROJ`, `NVME` and `REPO`. The repository is expected at
  `$PROJ/PlanningWithWorldModels`.

## Emulator

- Atari 2600 Tetris deals a fixed 16-piece cycle, the same in every game
  whatever the seed, no-ops or sticky actions. Agents trained on it
  (`atari_tetris`, and `custom_tetris` before `random_pieces`) memorise an
  opening rather than learn Tetris. `environments/tetris.py` randomises
  pieces by default.
- ALE 0.10's `getRAM()` returns the RAM from before `restoreState` or
  `reset_game` until the next frame is emulated. Code that restores a state and then reads RAM (as
  `environments/tetris.py` does on every step) must emulate a frame first.

## Delta

- Logging in needs a password and a Duo push, so an agent cannot reach Delta
  on its own. Ask the user to run `slurm/ssh_delta.sh` first; it keeps the
  connection open for 10 minutes, and `sync_from_delta.sh` reuses it.
- Do not submit, cancel or modify Slurm jobs without asking; they spend the
  `biny-delta-gpu` allocation.
- `#SBATCH -o/-e` lines cannot use variables, so log paths in job scripts are
  written out in full.

## Code style

- Python with 2-space indentation, single quotes, lines of about 80 columns,
  and no type hints. Match the surrounding code.
- Docstrings are short and say what a function is for; scripts start with a
  docstring giving example command lines. Every argparse flag has `help=`
  unless its name says it all.
- Comments and docstrings describe the code as it is, not how it changed.
  No "now", "used to", "moved from" or "instead of the old X"; that
  history belongs in commit messages.
- Import `world_model` before anything from `dreamerv3` or `embodied`:
  importing it puts the DreamerV3 checkout on `sys.path`. The `# noqa: E402`
  imports after that path setup are deliberate; do not reorder them.
- JAX: model code runs as `jax.jit(nj.pure(fn))` and is called with
  `seed=`; random keys come from the function `seeder()` returns. DreamerV3
  enables a host-to-device transfer guard, so index and reshape device arrays
  inside jitted code, and move inputs with `jax.device_put`.

## Checking changes

Say what was run and what was not. Useful checks:

- `python -m py_compile` on changed files.
- A short CPU training run: `python train.py --task <task> --configs debug
  --logdir <scratch dir> --run.steps 400`, with the `debug` preset's tiny
  model on the CPU.
- `python planning/<script>.py --help` from the repository root, which
  exercises the imports and path setup.
- Environments can be stepped directly, e.g. `environments.make('catch')`
  with `PYTHONPATH=.:third_party/dreamerv3`.

## Git

- Commit messages: a short imperative summary line, then a body when the
  change needs explaining.
- Work lands on `main` directly, without pull requests. Ask before pushing.
