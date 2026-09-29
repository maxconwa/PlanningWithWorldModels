#!/bin/bash
#SBATCH --job-name=tetris_400m
#SBATCH --account=biny-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=62g
#SBATCH --time=48:00:00
#SBATCH -o /projects/biny/mconway/logs/slurm-%j.out
#SBATCH -e /projects/biny/mconway/logs/slurm-%j.err

source /projects/biny/mconway/PlanningWithWorldModels/slurm/activate.sh
export PYTHONUNBUFFERED=1
cd $REPO/third_party/dreamerv3
nvidia-smi

python dreamerv3/main.py \
  --logdir $NVME/logdir/tetris_400m \
  --configs atari size400m \
  --task atari_tetris \
  --env.atari.actions needed \
  --env.atari.sticky False \
  --env.atari.gray False \
  --replay.size 3e5 \
  --run.steps 5e6 \
  --script train
