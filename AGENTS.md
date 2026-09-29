# Agent guide

Planning with DreamerV3 world models: agents are trained on NCSA Delta (Slurm),
then evaluated and searched over locally. `README.md` covers the layout and the
day-to-day commands; this file covers what an agent needs to work here safely.

## Layout

- `third_party/dreamerv3`: DreamerV3 submodule, pinned at `e3f0224`. That is
  the exact commit the checkpoints in `outputs/train/` were trained with, and
  the checkout on Delta is clean at it. Do not edit the submodule or move the
  pin without asking: checkpoints may stop loading.
- `third_party/open-dreamer`: open-dreamer submodule.
- `environments/`: custom training environments (`embodied.Env` subclasses),
  listed in `REGISTRY` in `environments/__init__.py`. They are trained with
  `python train.py --task custom_<name>`; `train.py` wraps DreamerV3's
  `main.py` and adds the `custom` suite without changing the submodule.
- `slurm/`: Delta job scripts, `activate.sh`, and the local helpers
  `ssh_delta.sh` and `sync_from_delta.sh`.
- `outputs/train/<run>/`: training runs (DreamerV3 logdirs, latest checkpoint
  only). `outputs/eval/<run>/`: evaluation and planning results for that run.
  `outputs/logs/`: Slurm logs.
- `planning/`: planning in trained world models. `world_model.py` loads a
  run (`build`, `warmup`) and resolves paths: `logdir_arg` finds a bare run
  name in `outputs/train/`, and `eval_path` puts a bare `--out` file name in
  `outputs/eval/<run>/`. `search.py` holds the planners (`make_search`). The
  scripts next to them import these as sibling modules, so they run as
  `python planning/<script>.py`. New planners go in `search.py`, and new
  planning scripts should use `logdir_arg` and `eval_path`.

## Storage

- Everything under `outputs/` is stored in Git LFS (`.gitattributes`), as are
  `*.pkl` files anywhere.
- GitHub rejects LFS files over 2 GiB. Checkpoints that large go in
  `.gitignore` (the tetris `agent.pkl` files already are, so those runs do not
  load from a fresh clone).
- Never commit replay buffers (`replay/`, `replay_old/`).

## Environments

- Local: the `dreamerv3` conda env (`~/miniconda3/envs/dreamerv3`) has JAX
  0.10.2, but the pinned DreamerV3 needs JAX 0.4.33 (see
  `third_party/dreamerv3/requirements.txt`). Building the agent fails there
  with `TypeError: jit() takes from 0 to 1 positional arguments`, so training
  and agent-loading scripts do not run locally until that env is fixed.
  Environment code alone (`environments/`) does run in it.
- Delta: `slurm/activate.sh` loads the `/projects/biny/mconway/dreamer-env`
  conda env and sets `PROJ`, `NVME` and `REPO`. The repo is expected to be
  cloned at `$PROJ/PlanningWithWorldModels`.

## Delta

- Logging in needs a password and a Duo push, so an agent cannot reach Delta
  on its own. Ask the user to run `slurm/ssh_delta.sh` first; it keeps the
  connection open for 10 minutes, and `sync_from_delta.sh` reuses it.
- Do not submit, cancel or modify Slurm jobs without asking; they spend the
  `biny-delta-gpu` allocation.
- `#SBATCH -o/-e` lines cannot use variables, so log paths in job scripts are
  written out in full.

## Code style

- Python with 2-space indentation, single quotes, and short docstrings that
  say what a function is for. Match the surrounding code.
- Scripts start with a docstring giving example command lines.
- Comments and docstrings describe the code as it is, not how it changed.
  No "now", "used to", "moved from" or "instead of the old X"; that
  history belongs in commit messages.
