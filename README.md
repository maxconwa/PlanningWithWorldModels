# PlanningWithWorldModels

Planning with DreamerV3 world models: training agents on NCSA Delta, then
running tree search over the learned models locally.

## Layout

- `third_party/dreamerv3`: DreamerV3 submodule, used both on Delta and locally.
  Clone with `git clone --recursive`, or run `git submodule update --init`.
- `third_party/open-dreamer`: open-dreamer submodule.
- `slurm/`: Delta job scripts and helpers.
- `environments/`: custom training environments for world models; train on
  one with `python train.py --task custom_<name>` (see
  `environments/README.md`).
- `outputs/`: everything produced on Delta, stored in Git LFS.
  - `outputs/train/<run>/`: a training run (config, metrics, latest checkpoint).
  - `outputs/eval/<run>/`: evaluation and planning results for that run.
  - `outputs/logs/`: Slurm `.out`/`.err` files.
- `planning/`: planning in a trained world model (tree search, evaluation,
  visualisation), e.g. `python planning/eval_model.py --logdir tetris
  --planner gumbel`. See `planning/README.md`.
- `jsonl_to_tensorboard.py`: turns a run's `metrics.jsonl` into TensorBoard
  logs.

## Training on Delta

One-time setup, on a login node:

    cd /projects/biny/mconway
    GIT_LFS_SKIP_SMUDGE=1 git clone --recursive https://github.com/maxconwa/PlanningWithWorldModels.git PlanningWithWorldModels

`slurm/activate.sh` loads the `dreamer-env` conda environment and sets `PROJ`,
`NVME` and `REPO`. Submit a job with:

    sbatch $REPO/slurm/tetris_400m.slurm

Runs write to `$NVME/logdir/<run>`, and Slurm output goes to
`/projects/biny/mconway/logs/slurm-<jobid>.{out,err}`.

## Copying a run to this machine

    slurm/ssh_delta.sh                        # log in once (password + Duo)
    slurm/sync_from_delta.sh tetris_400m      # copies into outputs/train/tetris_400m

The sync reuses the login connection for 10 minutes, so it does not ask for
Duo again. It copies the latest checkpoint only, skips the replay buffer, and
also copies the Slurm logs into `outputs/logs/`. Commit the result with
`git add outputs && git commit`; `.gitattributes` sends everything under
`outputs/` to LFS. GitHub rejects LFS files over 2 GiB, so list any larger
checkpoint in `.gitignore`.
