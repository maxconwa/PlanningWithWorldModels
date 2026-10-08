"""Block stacking with a Franka Panda, acting through end-effector deltas.

BatchStack steps a batch of worlds together on the GPU. Each episode samples
N blocks (2-4) and a goal order; the agent sees a workspace camera, a wrist
camera, the arm's state and the goal, and moves the gripper by up to 2 cm
and 0.2 rad of yaw per 20 Hz control step. Scalar diagnostics are log/ keys;
the full block state comes from privileged(). Stack is the embodied.Env
around it, which DreamerV3 and play.py use.
"""

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np

from environments import stack_scene as scene
from environments import stack_task as task
from environments.stack_control import HOLD, Controller
from environments.stack_sim import Sim

ARM_DIM = 19  # 7 joint angles, 7 velocities, gripper xyz, yaw, opening.
SETTLE = 10  # Physics steps (0.05 s) to let a new layout come to rest.


def arm_features(sim, control, d):
  """The arm observation: joint angles and velocities, gripper position,
  yaw and opening."""
  ee_pos, ee_yaw = control.ee(d)
  return jnp.concatenate([
      d.qpos[:, :7], d.qvel[:, :7], ee_pos, ee_yaw[:, None],
      sim.opening(d)[:, None]], -1).astype(jnp.float32)


def obs_space(size):
  image = elements.Space(np.uint8, tuple(size) + (3,))
  return {
      'workspace': image,
      'wrist': image,
      'arm': elements.Space(np.float32, (ARM_DIM,)),
      'goal': elements.Space(np.int32, (len(scene.COLORS),), 0, task.NONE + 1),
      'reward': elements.Space(np.float32),
      'is_first': elements.Space(bool),
      'is_last': elements.Space(bool),
      'is_terminal': elements.Space(bool),
      'log/prefix': elements.Space(np.int32),
      'log/n': elements.Space(np.int32),
      'log/success': elements.Space(np.int32),
      'log/sim_steps': elements.Space(np.int32),
  }


def act_space():
  return {
      'reset': elements.Space(bool),
      'action': elements.Space(np.float32, (5,), -1, 1),
  }


class BatchStack:

  def __init__(self, num_envs, size=(64, 64), n=0, prestack=0.3, seed=0):
    self.num_envs = num_envs
    self.fixed_n, self.prestack = int(n), float(prestack)
    # Building moves the model to the GPU, and compiling reads some of it
    # back. DreamerV3 forbids implicit transfers, so both happen here, ahead
    # of time; every step then moves its inputs and outputs explicitly.
    with jax.transfer_guard('allow'):
      self.sim = Sim(num_envs, size)
      self.control = Controller(self.sim)
      self.limits = jax.device_put(np.asarray(task.LIMITS, np.int32))
      self.hold = jax.device_put(np.asarray(HOLD, np.float32))
      self.state = self._init(jax.random.PRNGKey(seed))
      mask = jax.device_put(np.ones(num_envs, bool))
      action = jax.device_put(np.zeros((num_envs, 5), np.float32))
      self._restart = jax.jit(self._restart_impl).lower(
          self.state, mask).compile()
      self._advance = jax.jit(self._advance_impl).lower(
          self.state, action, mask).compile()
      self._privileged = jax.jit(self._privileged_impl).lower(
          self.state).compile()
    self.done = np.ones(num_envs, bool)

  def step(self, acts):
    """One control step for every world; ended or flagged worlds restart."""
    restart = np.asarray(acts['reset'], bool) | self.done
    mask = jax.device_put(restart)
    if restart.any():
      self.state = self._restart(self.state, mask)
    action = jax.device_put(np.asarray(acts['action'], np.float32))
    self.state, obs = self._advance(self.state, action, mask)
    obs = {k: np.asarray(v) for k, v in jax.device_get(obs).items()}
    self.done = obs['is_last'].copy()
    return obs

  def privileged(self):
    """The simulator's block state, which the agent does not observe."""
    out = jax.device_get(self._privileged(self.state))
    return {k: np.asarray(v) for k, v in out.items()}

  def _init(self, key):
    b = self.num_envs
    d = self.sim.make_data()
    zeros = jnp.zeros(b, jnp.int32)
    target, yaw = self.control.start(d)
    return dict(
        data=d, key=key, target=target, yaw=yaw,
        goal=jnp.full((b, len(scene.COLORS)), task.NONE, jnp.int32),
        n=zeros + 2, k=zeros, count=zeros, steps=zeros, sim_steps=zeros,
        episodes=zeros)

  def _restart_impl(self, state, mask):
    """New episodes for the worlds in `mask`; the others are untouched.
    A world's j-th start comes from the seed, the world and j alone, so it
    does not depend on when the other worlds' episodes end."""
    worlds = jnp.arange(self.num_envs)
    keys = jax.vmap(lambda w, e: jax.random.fold_in(
        jax.random.fold_in(state['key'], w), e))(worlds, state['episodes'])
    n, goal, pos, yaws = jax.vmap(task.sample_start, (0, None, None))(
        keys, self.fixed_n, self.prestack)
    fresh = self.sim.set_blocks(self.sim.home(state['data']), pos, yaws)
    fresh = self.sim.physics(fresh, fresh.ctrl, SETTLE)
    d = state['data']
    m = mask[:, None]
    d = self.sim.forward(d.replace(
        qpos=jnp.where(m, fresh.qpos, d.qpos),
        qvel=jnp.where(m, fresh.qvel, d.qvel),
        ctrl=jnp.where(m, fresh.ctrl, d.ctrl)))
    target, yaw = self.control.start(d)

    def pick(new, old):
      return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)),
                       new, old)

    return dict(
        state, data=d, episodes=state['episodes'] + mask,
        target=pick(target, state['target']), yaw=pick(yaw, state['yaw']),
        goal=pick(goal, state['goal']), n=pick(n, state['n']))

  def _advance_impl(self, state, action, first):
    """One control step. The first step of a new episode holds still and
    reports the starting state with no reward."""
    action = jnp.where(first[:, None], self.hold, action)
    d = state['data']
    ctrl, target, yaw = self.control.act(
        d, state['target'], state['yaw'], action)
    d = self.sim.physics(d, ctrl, scene.SUBSTEPS)
    workspace, wrist, d = self.sim.render(d)
    pos, opening = self.sim.block_pos(d), self.sim.opening(d)
    held = jax.vmap(task.held_mask)(pos, self.sim.site_pos(d), opening)
    k = jax.vmap(task.tower_prefix)(state['goal'], state['n'], pos, held)
    reward, success, count = jax.vmap(task.progress)(
        state['k'], state['count'], k, state['n'],
        opening > scene.OPEN_WIDTH)
    reward = jnp.where(first, 0.0, reward)
    success = success & ~first
    count = jnp.where(first, 0, count)
    steps = jnp.where(first, 0, state['steps'] + 1)
    sim_steps = state['sim_steps'] + jnp.where(first, 0, 1)
    last = success | (steps >= self.limits[state['n']])
    arm = arm_features(self.sim, self.control, d)
    obs = {
        'workspace': workspace, 'wrist': wrist,
        'arm': arm, 'goal': state['goal'],
        'reward': reward.astype(jnp.float32),
        'is_first': first, 'is_last': last, 'is_terminal': success,
        'log/prefix': k.astype(jnp.int32), 'log/n': state['n'],
        'log/success': success.astype(jnp.int32),
        'log/sim_steps': sim_steps.astype(jnp.int32),
    }
    state = dict(state, data=d, target=target, yaw=yaw, k=k, count=count,
                 steps=steps, sim_steps=sim_steps)
    return state, obs

  def _privileged_impl(self, state):
    d = state['data']
    colors = jnp.arange(len(scene.COLORS))
    present = (colors[None, :, None] == state['goal'][:, None, :]).any(-1)
    pos = self.sim.block_pos(d)
    blocks = jnp.concatenate([
        present[..., None].astype(jnp.float32), pos,
        self.sim.block_quat(d), self.sim.block_vel(d)], -1)
    blocks = blocks * present[..., None]
    held = jax.vmap(task.held_mask)(
        pos, self.sim.site_pos(d), self.sim.opening(d)) & present
    held = jnp.where(held.any(-1), jnp.argmax(held, -1), -1)
    return {'blocks': blocks, 'held': held.astype(jnp.int32)}


class Stack(embodied.Env):

  BATCHED = True
  OPTIONS = 'stack'

  def __init__(self, task_name, batch=1, size=(64, 64), n=0, prestack=0.3,
               seed=None):
    del task_name
    self.batch, self.size = int(batch), tuple(size)
    self.options = dict(n=n, prestack=prestack)
    self.seed = int(np.random.SeedSequence().entropy % 2 ** 31
                    if seed is None else seed)
    self._env = None

  @property
  def env(self):
    if self._env is None:
      self._env = BatchStack(self.batch, self.size, seed=self.seed,
                             **self.options)
    return self._env

  @property
  def obs_space(self):
    return obs_space(self.size)

  @property
  def act_space(self):
    return act_space()

  def step(self, action):
    assert self.batch == 1, 'Use step_batch() for more than one world.'
    acts = {k: np.asarray(v)[None] for k, v in action.items()}
    return {k: v[0] for k, v in self.env.step(acts).items()}

  def step_batch(self, acts):
    return self.env.step(acts)

  def privileged(self):
    return self.env.privileged()
