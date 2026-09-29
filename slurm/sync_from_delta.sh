#!/bin/bash
# Copy a training run from Delta into outputs/train/, keeping only its latest
# checkpoint and skipping the replay buffer. Also copies the Slurm logs into
# outputs/logs/.
#
#   slurm/sync_from_delta.sh tetris_400m            # into outputs/train/tetris_400m
#   slurm/sync_from_delta.sh tetris_400m tetris_5m  # into outputs/train/tetris_5m
#   slurm/sync_from_delta.sh                        # Slurm logs only
set -euo pipefail

outputs=$(dirname "$0")/../outputs
ssh="ssh -o ControlMaster=auto -o ControlPath=~/.ssh/cm-%C -o ControlPersist=10m"
host=mconway@login.delta.ncsa.illinois.edu

mkdir -p "$outputs/logs"
rsync -av -e "$ssh" "$host:/projects/biny/mconway/logs/" "$outputs/logs/"

if [ $# -ge 1 ]; then
  run=$1
  dest=$outputs/train/${2:-$run}
  src=/work/nvme/biny/mconway/logdir/$run
  latest=$($ssh $host cat "$src/ckpt/latest")
  mkdir -p "$dest/ckpt"
  rsync -av -e "$ssh" --exclude 'replay*/' --exclude ckpt/ "$host:$src/" "$dest/"
  rsync -av -e "$ssh" "$host:$src/ckpt/latest" "$host:$src/ckpt/$latest" "$dest/ckpt/"
fi
