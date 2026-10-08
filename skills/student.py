"""Distil a skill's state teacher into an image student with DAgger.

  python skills/student.py --skill pick
  python skills/student.py --skill place --iterations 30

The student sees what the option environment will give it: both cameras,
the arm, the target and the time fraction. Each iteration plays the
batched skill environment with the student acting in each world (the
teacher acts with probability 0.5 ** iteration instead), labels every
visited state with the teacher's action, and trains the student on all
labels so far. Writes outputs/train/skill_<skill>_student/.
"""

import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
# Run as a script: put the repository and DreamerV3 (for environments/) on
# the path, and let MuJoCo Warp allocate GPU memory beside JAX.
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402


class Student(nn.Module):
  """Both cameras through a small CNN, joined with the proprioception."""

  @nn.compact
  def __call__(self, images, proprio):
    x = images.astype(jnp.float32) / 255.0 - 0.5
    for depth in (32, 64, 128, 256):
      x = nn.relu(nn.Conv(depth, (4, 4), strides=(2, 2))(x))
    x = x.reshape(x.shape[0], -1)
    x = jnp.concatenate([x, proprio], -1)
    x = nn.relu(nn.Dense(512)(x))
    x = nn.relu(nn.Dense(256)(x))
    return nn.Dense(5)(x)


def _inputs(obs):
  images = jnp.concatenate([obs['workspace'], obs['wrist']], -1)
  return images, obs['proprio']


def act(params, obs):
  """Actions in [-1, 1]."""
  return jnp.tanh(Student().apply(params, *_inputs(obs)))


def _loss(params, images, proprio, target):
  out = Student().apply(params, images, proprio)
  return jnp.mean((jnp.tanh(out) - target) ** 2)


def dagger(skill, teacher, out, iterations=20, num_envs=128, rollout=300,
           stride=2, train_steps=2000, capacity=150_000, batch=256, seed=0):
  from skills.env import PROPRIO_DIM, SkillEnv
  out = pathlib.Path(out)
  out.mkdir(parents=True, exist_ok=True)
  env = SkillEnv(skill, num_envs, cameras=True)
  key = jax.random.PRNGKey(seed)
  key, init_key, reset_key = jax.random.split(key, 3)
  state = jax.jit(env.reset)(jax.random.split(reset_key, num_envs))
  params = Student().init(init_key, *_inputs(state.obs))
  optimizer = optax.adam(3e-4)
  opt_state = optimizer.init(params)

  @jax.jit
  def collect(state, params, key, beta):
    labels = teacher(state.obs['state'])
    mine = act(params, state.obs)
    use_teacher = jax.random.uniform(key, (num_envs,)) < beta
    action = jnp.where(use_teacher[:, None], labels, mine)
    return env.step(state, action), labels

  @jax.jit
  def update(params, opt_state, images, proprio, target):
    loss, grads = jax.value_and_grad(_loss)(params, images, proprio, target)
    updates, opt_state = optimizer.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

  images = np.zeros((capacity, 64, 64, 6), np.uint8)
  proprio = np.zeros((capacity, PROPRIO_DIM), np.float32)
  targets = np.zeros((capacity, 5), np.float32)
  size, cursor, steps = 0, 0, 0
  rng = np.random.default_rng(seed)
  log = (out / 'metrics.jsonl').open('w')
  start = time.time()
  for it in range(iterations):
    beta = 0.5 ** it
    for t in range(rollout):
      key, sub = jax.random.split(key)
      obs = state.obs
      state, labels = collect(state, params, sub, beta)
      steps += num_envs
      if t % stride:
        continue
      ims, prop = jax.device_get(_inputs(obs))
      slots = (cursor + np.arange(num_envs)) % capacity
      images[slots], proprio[slots] = ims, prop
      targets[slots] = np.asarray(jax.device_get(labels))
      cursor = int((cursor + num_envs) % capacity)
      size = min(size + num_envs, capacity)
    losses = []
    for _ in range(train_steps):
      idx = rng.integers(0, size, batch)
      params, opt_state, loss = update(
          params, opt_state, jax.device_put(images[idx]),
          jax.device_put(proprio[idx]), jax.device_put(targets[idx]))
      losses.append(float(loss))
    row = dict(iteration=it, beta=beta, samples=size, control_steps=steps,
               loss=float(np.mean(losses[-50:])),
               seconds=time.time() - start)
    log.write(json.dumps(row) + '\n')
    log.flush()
    print(json.dumps(row), flush=True)
  with (out / 'params.pkl').open('wb') as f:
    pickle.dump(jax.device_get(params), f)
  config = dict(skill=skill, iterations=iterations, num_envs=num_envs,
                rollout=rollout, stride=stride, train_steps=train_steps,
                capacity=capacity, batch=batch, seed=seed,
                control_steps=steps, seconds=time.time() - start)
  (out / 'config.json').write_text(json.dumps(config, indent=2))
  return config


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--skill', choices=['pick', 'place'], required=True)
  parser.add_argument('--iterations', type=int, default=20)
  parser.add_argument('--num-envs', type=int, default=128)
  parser.add_argument('--seed', type=int, default=0)
  args = parser.parse_args()
  from skills.policy import load_teacher
  runs = REPO / 'outputs' / 'train'
  teacher = load_teacher(runs / f'skill_{args.skill}_teacher')
  config = dagger(args.skill, teacher, runs / f'skill_{args.skill}_student',
                  args.iterations, args.num_envs, seed=args.seed)
  print(f"{config['control_steps']:,} control steps in "
        f"{config['seconds'] / 60:.0f} min")


if __name__ == '__main__':
  main()
