import threading

import jax
import numpy as np
import pytest

import environments
from environments.stack import BatchStack, obs_space
from environments.stack_scene import HALF
from environments.stack_task import HOLD_STEPS, LIMITS

B = 4


def acts(action=(0, 0, 0, 0, -1), reset=False, batch=B):
  return {'reset': np.full(batch, reset),
          'action': np.tile(np.asarray(action, np.float32), (batch, 1))}


def check_obs(obs, batch=B):
  for key, space in obs_space((64, 64)).items():
    value = np.asarray(obs[key])
    assert value.shape == (batch,) + space.shape, key
    assert value.dtype == space.dtype, key
  assert np.isfinite(obs['arm']).all()


def teleport(env, world, placements):
  """Move blocks of one world: {colour: (x, y, z)}; others stay put."""
  pos = np.array(jax.device_get(env.sim.block_pos(env.state['data'])))
  for c, xyz in placements.items():
    pos[world, c] = xyz
  env.state['data'] = env.sim.set_blocks(
      env.state['data'], jax.device_put(pos.astype(np.float32)),
      jax.device_put(np.zeros((env.num_envs, 4), np.float32)))


def build_tower(env, world=0):
  """Teleport the world's episode blocks into its goal tower."""
  goal = np.asarray(jax.device_get(env.state['goal']))[world]
  n = int(jax.device_get(env.state['n'])[world])
  teleport(env, world, {int(c): (0.5, 0.15, HALF + 2 * HALF * level)
                        for level, c in enumerate(goal[:n])})


@pytest.fixture(scope='module')
def env():
  return BatchStack(B, seed=0)


def test_first_observation(env):
  obs = env.step(acts(reset=True))
  check_obs(obs)
  assert obs['is_first'].all() and not obs['is_last'].any()
  assert (obs['reward'] == 0).all()
  for i in range(B):
    n = int(obs['log/n'][i])
    assert (obs['goal'][i][:n] < 4).all() and (obs['goal'][i][n:] == 4).all()


def test_step_observation(env):
  env.step(acts(reset=True))
  obs = env.step(acts((0.5, 0, 0, 0, -1)))
  check_obs(obs)
  assert not obs['is_first'].any()


def test_reset_starts_at_rest():
  env = BatchStack(64, seed=1, prestack=0.5)
  env.step(acts(reset=True, batch=64))
  blocks = env.privileged()['blocks']
  present = blocks[..., 0] > 0
  assert np.abs(blocks[..., 8:11][present]).max() < 0.02


def test_privileged_marks_present_blocks(env):
  obs = env.step(acts(reset=True))
  blocks = env.privileged()['blocks']
  assert blocks.shape == (B, 4, 11)
  for i in range(B):
    present = set(np.flatnonzero(blocks[i, :, 0]).tolist())
    n = int(obs['log/n'][i])
    assert present == set(obs['goal'][i][:n].tolist())


def test_building_the_tower_succeeds():
  env = BatchStack(2, seed=3, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=2))
  build_tower(env, 0)
  total = 0.0
  for _ in range(HOLD_STEPS + 5):
    obs = env.step(acts(batch=2))
    total += float(obs['reward'][0])
    if obs['is_last'][0]:
      break
  assert obs['is_last'][0] and obs['is_terminal'][0]
  assert int(obs['log/success'][0]) == 1
  assert total == pytest.approx(2.0)  # Prefix 1 -> 2, plus success.
  assert not obs['is_last'][1]


def test_toppling_gives_negative_reward():
  env = BatchStack(1, seed=4, n=3, prestack=0.0)
  env.step(acts(reset=True, batch=1))
  build_tower(env, 0)
  env.step(acts(batch=1))
  top = int(np.asarray(jax.device_get(env.state['goal']))[0, 2])
  teleport(env, 0, {top: (0.6, -0.1, HALF)})
  obs = env.step(acts(batch=1))
  assert float(obs['reward'][0]) == pytest.approx(-1.0)
  assert not obs['is_last'][0]


def test_time_limit_ends_episode():
  env = BatchStack(1, seed=5, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=1))
  for _ in range(LIMITS[2]):
    obs = env.step(acts(batch=1))
  assert obs['is_last'][0] and not obs['is_terminal'][0]
  assert env.step(acts(batch=1))['is_first'][0]


def test_worlds_reset_independently():
  env = BatchStack(2, seed=6, n=2, prestack=0.0)
  env.step(acts(reset=True, batch=2))
  for _ in range(5):
    env.step(acts(batch=2))
  before = int(env.step(acts(batch=2))['log/sim_steps'][1])
  flagged = acts(batch=2)
  flagged['reset'][0] = True
  obs = env.step(flagged)
  assert obs['is_first'][0] and not obs['is_first'][1]
  assert int(obs['log/sim_steps'][1]) == before + 1


def test_same_seed_same_start():
  a, b = BatchStack(2, seed=7), BatchStack(2, seed=7)
  oa = a.step(acts(reset=True, batch=2))
  ob = b.step(acts(reset=True, batch=2))
  np.testing.assert_array_equal(oa['goal'], ob['goal'])
  np.testing.assert_allclose(a.privileged()['blocks'],
                             b.privileged()['blocks'], atol=1e-5)


def test_runs_under_transfer_guard():
  with jax.transfer_guard('disallow'):
    env = BatchStack(2, seed=8)
    env.step(acts(reset=True, batch=2))
    env.step(acts((0.3, 0, 0, 0, -1), batch=2))
    env.privileged()


def test_single_world_env():
  env = environments.make('stack', seed=0)
  assert env.obs_space['workspace'].shape == (64, 64, 3)
  obs = env.step({'reset': True, 'action': np.zeros(5, np.float32)})
  assert obs['is_first'] and obs['workspace'].shape == (64, 64, 3)
  obs = env.step({'reset': False, 'action': np.zeros(5, np.float32)})
  assert not obs['is_first']


def test_steps_while_another_thread_uses_the_gpu():
  """DreamerV3 trains and prefetches batches in other threads while the
  environment steps."""
  env = BatchStack(2, seed=9)
  matmul = jax.jit(lambda x: (x @ x) @ x)
  stop = threading.Event()

  def busy():
    while not stop.is_set():
      x = jax.device_put(np.ones((2048, 2048), np.float32))
      jax.block_until_ready(matmul(x))

  thread = threading.Thread(target=busy)
  thread.start()
  try:
    env.step(acts(reset=True, batch=2))
    for _ in range(30):
      obs = env.step(acts((0.3, 0, 0, 0, -1), batch=2))
  finally:
    stop.set()
    thread.join()
  assert np.isfinite(obs['arm']).all()


def test_each_worlds_starts_follow_from_the_seed():
  """A world's j-th start is the same whenever the other worlds reset."""
  a, b = BatchStack(2, seed=10), BatchStack(2, seed=10)
  a.step(acts(reset=True, batch=2))
  b.step(acts(reset=True, batch=2))
  only_world_0 = acts(batch=2)
  only_world_0['reset'][0] = True
  for _ in range(3):
    a.step(only_world_0)
  only_world_1 = acts(batch=2)
  only_world_1['reset'][1] = True
  oa, ob = a.step(only_world_1), b.step(only_world_1)
  np.testing.assert_array_equal(oa['goal'][1], ob['goal'][1])
  np.testing.assert_allclose(a.privileged()['blocks'][1],
                             b.privileged()['blocks'][1], atol=1e-5)


def test_reset_keeps_the_sampled_layout():
  """After the settle the blocks are where their starts put them: no
  pre-built stack has fallen. GPU physics is not deterministic, so this
  bounds the rate: box geoms moved 6-10 of 768 worlds, cube meshes 0-1."""
  from environments.stack_task import sample_start
  moved = 0
  for seed in (11, 13, 14):
    env = BatchStack(256, seed=seed, prestack=1.0)
    env.step(acts(reset=True, batch=256))
    base = jax.random.PRNGKey(seed)
    keys = jax.numpy.stack([jax.random.fold_in(jax.random.fold_in(base, w), 0)
                            for w in range(256)])
    _, _, want, _ = jax.vmap(sample_start, (0, None, None))(keys, 0, 1.0)
    blocks = env.privileged()['blocks']
    far = np.linalg.norm(blocks[..., 1:4] - np.asarray(want), axis=-1) > 0.005
    moved += int((far & (blocks[..., 0] > 0)).any(-1).sum())
  assert moved <= 2, f'{moved} of 768 worlds start away from their layout'


def test_settled_stacks_stay_put():
  """Pre-built stacks do not topple on their own over 5 s. GPU physics is
  not deterministic, so this bounds the rate: box geoms toppled 6-10 of 768
  worlds, cube meshes 0-1."""
  moved = 0
  for seed in (21, 22, 23):
    env = BatchStack(256, seed=seed, prestack=1.0)
    env.step(acts(reset=True, batch=256))
    blocks = env.privileged()['blocks']
    before, present = blocks[..., 1:4], blocks[..., 0] > 0
    d = env.state['data']
    step = jax.jit(env.sim.physics, static_argnums=2)
    for _ in range(25):
      d = step(d, d.ctrl, 40)
    after = np.asarray(jax.device_get(env.sim.block_pos(d)))
    far = np.linalg.norm(after - before, axis=-1) > 0.005
    moved += int((far & present).any(-1).sum())
  assert moved <= 2, f'{moved} of 768 worlds had a block move'


def test_a_reset_costs_little_more_than_a_step():
  """One world restarting must not hold up the batch for long: the settle
  simulates every world. A 60-step settle made it 5x a normal step."""
  import time
  env = BatchStack(16, seed=12)
  env.step(acts(reset=True, batch=16))
  normal, flagged = acts(batch=16), acts(batch=16)
  flagged['reset'][0] = True
  for batch in (normal, flagged, normal):
    env.step(batch)

  def timed(batch):
    start = time.time()
    for _ in range(10):
      env.step(batch)
    return time.time() - start

  assert timed(flagged) < 3 * timed(normal)
