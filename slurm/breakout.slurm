#!/bin/bash
#SBATCH --job-name=dreamer_breakout
#SBATCH --account=biny-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH -o /projects/biny/mconway/logs/slurm-%j.out
#SBATCH -e /projects/biny/mconway/logs/slurm-%j.err

source /projects/biny/mconway/PlanningWithWorldModels/slurm/activate.sh
export PYTHONUNBUFFERED=1
cd $REPO/third_party/dreamerv3
nvidia-smi

python dreamerv3/main.py \
  --logdir $NVME/logdir/breakout_eval10 \
  --configs atari100k \
  --task atari100k_breakout \
  --script train_eval \
  --run.eval_eps 10
