"""Batched Brax environments for the pick and place skills.

Each world starts from the stacking start distribution: for pick, the arm
at home and a clear block to pick; for place, a block already held above
the others and a target for it (a clear block or the table). The policy
moves the arm through the raw 5-D action. The environment ends a skill on
the step its success condition holds, or after tasks.EPISODE steps. A world
whose episode ended restarts within the same step,
with its own new start (Brax's AutoResetWrapper would replay its first one),
so PPO runs with wrap_env=False and the bookkeeping in brax_util.
"""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State

from environments import stack_scene as scene
from environments.stack import arm_features
from environments.stack_control import Controller, rot_z
from environments.stack_sim import Sim
from environments.stack_task import held_mask
from skills import tasks
from skills.brax_util import episode_info, track_episode

STATE_DIM = 85
PROPRIO_DIM = 25
ACTION_DIM = 5
SETTLE = 10  # Physics steps after a new start.
METRICS = ('success',)


class SkillEnv(Env):

  def __init__(self, skill, num_envs, cameras=False, prestack=0.3,
               stacked=0.0, size=(64, 64)):
    assert skill in ('pick', 'place'), skill
    self.skill, self.num_envs, self.cameras = skill, num_envs, cameras
    self.prestack, self.stacked = prestack, stacked
    self.sim = Sim(num_envs, size, cameras=cameras)
    self.control = Controller(self.sim)

  @property
  def observation_size(self):
    size = {'state': (STATE_DIM,)}
    if self.cameras:
      image = self.sim.size + (3,)
      size.update(workspace=image, wrist=image, proprio=(PROPRIO_DIM,))
    return size

  @property
  def action_size(self):
    return ACTION_DIM

  @property
  def backend(self):
    return 'mjx'

  def reset(self, rng):
    d = self.sim.make_data()
    d, task = self._fresh(d, rng)
    target, yaw = self.control.start(d)
    phi, _ = self._progress(d, task)
    zeros = jnp.zeros(self.num_envs)
    info = dict(episode_info(self.num_envs, METRICS), rng=rng, task=task,
                ee_target=target, ee_yaw=yaw, phi=phi, steps=zeros)
    obs, d = self._obs(d, info)
    metrics = {name: zeros for name in METRICS}
    return State(d, obs, zeros, zeros, metrics, info)

  def step(self, state, action):
    info = state.info
    a = jnp.clip(action, -1.0, 1.0)
    d = state.pipeline_state
    ctrl, target, yaw = self.control.act(
        d, info['ee_target'], info['ee_yaw'], a)
    d = self.sim.physics(d, ctrl, scene.SUBSTEPS)
    steps = info['steps'] + 1
    phi, success = self._progress(d, info['task'])
    timeout = steps >= tasks.EPISODE
    reward = phi - info['phi'] + jnp.where(success, tasks.SUCCESS_BONUS, 0.0)
    done = success | timeout
    metrics = {'success': success.astype(jnp.float32)}
    info = track_episode(info, reward, done, timeout & ~success, metrics)
    info = dict(info, ee_target=target, ee_yaw=yaw, phi=phi, steps=steps)
    d, info = jax.lax.cond(
        done.any(), lambda: self._restart(d, info, done), lambda: (d, info))
    obs, d = self._obs(d, info)
    return State(d, obs, reward.astype(jnp.float32),
                 done.astype(jnp.float32), metrics, info)

  def held(self, state):
    d = state.pipeline_state
    return jax.vmap(held_mask)(
        self.sim.block_pos(d), self.sim.site_pos(d),
        self.sim.opening(d)) & state.info['task']['present']

  def _fresh(self, d, keys):
    """New starts for every world, settled."""
    d = self.sim.home(d)
    if self.skill == 'pick':
      present, pos, yaws, target = jax.vmap(
          tasks.pick_start, (0, None, None))(keys, self.prestack, self.stacked)
      held_block = target
    else:
      present, pos, yaws, held_block, target, site, yaw = jax.vmap(
          tasks.place_start, (0, None))(keys, self.prestack)
      mats = jax.vmap(rot_z)(yaw) @ self.control.home_mat
      q = jax.vmap(lambda qpos, p, m: self.control.solve(
          qpos, p, m, iters=30))(d.qpos, site, mats)
      fingers = jnp.full((self.num_envs, 2), scene.HALF)
      d = d.replace(
          qpos=d.qpos.at[:, :7].set(q).at[:, 7:9].set(fingers),
          ctrl=d.ctrl.at[:, :7].set(q).at[:, 7].set(scene.CLOSED))
    d = self.sim.set_blocks(d, pos, yaws)
    d = self.sim.physics(d, d.ctrl, SETTLE)
    task = {'present': present, 'target': target.astype(jnp.int32),
            'held_block': held_block.astype(jnp.int32)}
    return d, task

  def _restart(self, d, info, mask):
    """New starts for the worlds in `mask`; the others are untouched."""
    # Only restarted worlds draw a new key, so a world's sequence of starts
    # does not depend on when the others' episodes end.
    split = jax.vmap(jax.random.split)(info['rng'])
    rng = jnp.where(mask[:, None], split[:, 0], info['rng'])
    keys = split[:, 1]
    fresh, task = self._fresh(d, keys)
    m = mask[:, None]
    d = self.sim.forward(d.replace(
        qpos=jnp.where(m, fresh.qpos, d.qpos),
        qvel=jnp.where(m, fresh.qvel, d.qvel),
        ctrl=jnp.where(m, fresh.ctrl, d.ctrl)))

    def pick(new, old):
      return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)),
                       new, old)

    task = jax.tree.map(pick, task, info['task'])
    target, yaw = self.control.start(d)
    phi, _ = self._progress(d, task)
    return d, dict(
        info, rng=rng, task=task,
        ee_target=pick(target, info['ee_target']),
        ee_yaw=pick(yaw, info['ee_yaw']), phi=pick(phi, info['phi']),
        steps=jnp.where(mask, 0.0, info['steps']))

  def _progress(self, d, task):
    pos, site = self.sim.block_pos(d), self.sim.site_pos(d)
    opening = self.sim.opening(d)
    held = jax.vmap(held_mask)(pos, site, opening) & task['present']
    if self.skill == 'pick':
      return jax.vmap(tasks.pick_progress)(
          pos, task['present'], held, site, task['target'])
    return jax.vmap(tasks.place_progress)(
        pos, held, site, opening, task['held_block'], task['target'])

  def _obs(self, d, info):
    task = info['task']
    arm = arm_features(self.sim, self.control, d)
    pos, site = self.sim.block_pos(d), self.sim.site_pos(d)
    present = task['present'][..., None].astype(jnp.float32)
    blocks = jnp.concatenate([
        present, pos - site[:, None], pos, self.sim.block_quat(d),
        self.sim.block_vel(d)], -1) * present
    held = jax.vmap(held_mask)(pos, site, self.sim.opening(d))
    held = (held & task['present']).astype(jnp.float32)
    target = jax.nn.one_hot(task['target'], 5)
    time = (info['steps'] / tasks.EPISODE)[:, None]
    obs = {'state': jnp.concatenate([
        arm, blocks.reshape(self.num_envs, -1), target, held, time], -1)}
    if self.cameras:
      workspace, wrist, d = self.sim.render(d)
      obs.update(workspace=workspace, wrist=wrist,
                 proprio=jnp.concatenate([arm, target, time], -1))
    return obs, d
