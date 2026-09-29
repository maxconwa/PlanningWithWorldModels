module reset
module load miniforge3-python
eval "$(conda shell.bash hook)"
conda activate /projects/biny/mconway/dreamer-env
export PROJ=/projects/biny/mconway
export NVME=/work/nvme/biny/mconway
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_COMPILATION_CACHE_DIR=/work/nvme/biny/mconway/jax-cache
export REPO=$PROJ/PlanningWithWorldModels
