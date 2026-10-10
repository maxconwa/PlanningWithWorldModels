"""Block stacking through options: pick(colour), place_on(colour | table).

BatchStackOptions wraps BatchStack. Options 0-3 pick a colour, 4-7 place
the held block on a colour and 8 on the table. A valid option runs its
trained image skill on the raw action space until the skill's success
condition holds, for at most SKILL_STEPS control steps; an invalid one holds
still for INVALID_STEPS. A place that completes the tower holds still until
the episode's success is decided. Every option ends when the episode does.
Worlds whose option has ended are frozen until every world's has. The
observation is the last control step's, with the reward summed over the
option and log/duration its control steps. StackOptions is the embodied.Env
(custom_stack_options).
"""

import pathlib

import elements
import jax
import jax.numpy as jnp
import numpy as np

from environments import stack
from environments import stack_scene as scene
from environments.stack_task import held_mask
from skills import tasks
from skills.env import student_proprio
from skills.policy import student_params
from skills.student import act as student_act

SKILLS = pathlib.Path(__file__).resolve().parent.parent / 'outputs' / 'train'
COLOURS = len(scene.COLORS)
NUM_OPTIONS = 2 * COLOURS + 1
SKILL_STEPS = tasks.EPISODE
INVALID_STEPS = 10
OBS = ('workspace', 'wrist', 'arm')


def option_mask(pos, present, held):
  """Which options are valid in one world."""
  holding = held.any()
  clear = tasks.clear_mask(pos, present & ~held)
  return jnp.concatenate([clear & ~holding, clear & holding, holding[None]])


def obs_space(size):
  return {**stack.obs_space(size),
          'log/duration': elements.Space(np.int32)}


def act_space():
  return {
      'reset': elements.Space(bool),
      'action': elements.Space(np.int32, (), 0, NUM_OPTIONS),
  }


def _pick(mask, new, old):
  return jnp.where(mask.reshape(mask.shape + (1,) * (new.ndim - 1)), new, old)


class BatchStackOptions:

  def __init__(self, num_envs, size=(64, 64), n=0, prestack=0.3, seed=0,
               skills=None):
    self.num_envs, self.size = num_envs, tuple(size)
    self.base = stack.BatchStack(num_envs, size, n, prestack, seed)
    skills = pathlib.Path(skills or SKILLS)
    b = num_envs
    with jax.transfer_guard('allow'):
      self.params = (student_params(skills / 'skill_pick_student'),
                     student_params(skills / 'skill_place_student'))
      self.obs = {k: jax.device_put(np.zeros(
          (b,) + v.shape, v.dtype)) for k, v in stack.obs_space(size).items()}
      option = jax.device_put(np.zeros(b, np.int32))
      mask = jax.device_put(np.ones(b, bool))
      self._begin = jax.jit(self._begin_impl).lower(
          self.base.state, option, mask).compile()
      opt = self._begin(self.base.state, option, mask)
      self._tick = jax.jit(self._tick_impl).lower(
          self.base.state, opt, self.obs, mask, self.params).compile()
      self._mask = jax.jit(self._mask_impl).lower(self.base.state).compile()
      self.no_first = jax.device_put(np.zeros(b, bool))
    self.done = np.ones(b, bool)

  @property
  def state(self):
    return self.base.state

  @state.setter
  def state(self, value):
    self.base.state = value

  def step(self, acts):
    """One option in every world; ended or flagged worlds restart and
    report their first observation instead."""
    restart = np.asarray(acts['reset'], bool) | self.done
    first = jax.device_put(restart)
    if restart.any():
      self.state = self.base._restart(self.state, first)
    option = jax.device_put(np.asarray(acts['action'], np.int32))
    opt = self._begin(self.state, option, first)
    self.state, opt, self.obs, running = self._tick(
        self.state, opt, self.obs, first, self.params)
    while bool(jax.device_get(running)):
      self.state, opt, self.obs, running = self._tick(
          self.state, opt, self.obs, self.no_first, self.params)
    out = dict(self.obs, reward=opt['reward'],
               **{'log/duration': opt['steps']})
    out = {k: np.asarray(v) for k, v in jax.device_get(out).items()}
    self.done = out['is_last'].copy()
    return out

  def privileged(self):
    out = dict(self.base.privileged())
    out['option_mask'] = np.asarray(jax.device_get(self._mask(self.state)))
    return out

  def _blocks(self, state):
    sim, d = self.base.sim, state['data']
    pos, site, opening = sim.block_pos(d), sim.site_pos(d), sim.opening(d)
    colours = jnp.arange(COLOURS)
    present = (colours[None, :, None] == state['goal'][:, None, :]).any(-1)
    held = jax.vmap(held_mask)(pos, site, opening) & present
    return pos, site, opening, present, held

  def _mask_impl(self, state):
    pos, _, _, present, held = self._blocks(state)
    return jax.vmap(option_mask)(pos, present, held)

  def _begin_impl(self, state, option, first):
    pos, _, _, present, held = self._blocks(state)
    mask = jax.vmap(option_mask)(pos, present, held)
    valid = jnp.take_along_axis(mask, option[:, None], 1)[:, 0]
    pick = option < COLOURS
    zeros = jnp.zeros(self.num_envs, jnp.int32)
    return dict(
        valid=valid, present=present, pick=pick,
        target=jnp.where(pick, option, option - COLOURS).astype(jnp.int32),
        held_block=jnp.argmax(held, -1).astype(jnp.int32),
        running=~first, skill_done=jnp.zeros(self.num_envs, bool),
        steps=zeros, reward=jnp.zeros(self.num_envs, jnp.float32))

  def _tick_impl(self, state, opt, obs, first, params):
    """One control step for the worlds whose option runs and the worlds
    starting an episode; the others are frozen."""
    d = state['data']
    student_obs = dict(
        workspace=obs['workspace'], wrist=obs['wrist'],
        proprio=student_proprio(obs['arm'], opt['target'], opt['steps']))
    skill = jnp.where(opt['pick'][:, None],
                      student_act(params[0], student_obs),
                      student_act(params[1], student_obs))
    # Holding still keeps the gripper as it is, so a held block stays held.
    closed = d.ctrl[:, 7] < scene.OPEN / 2
    still = jnp.zeros((self.num_envs, 5)).at[:, 4].set(
        jnp.where(closed, 1.0, -1.0))
    wait = ~opt['valid'] | opt['skill_done']
    action = jnp.where(wait[:, None], still, skill)
    new, new_obs = self.base._advance_impl(state, action, first)
    update = opt['running'] | first
    state = self._select(update, new, state)
    obs = {k: _pick(update, new_obs[k], obs[k]) for k in obs}

    ran = opt['running']
    steps = opt['steps'] + ran
    reward = opt['reward'] + jnp.where(ran, new_obs['reward'], 0.0)
    pos, site, opening, _, held = self._blocks(state)
    present, target = opt['present'], opt['target']
    _, picked = jax.vmap(tasks.pick_progress)(
        pos, present, held, site, jnp.minimum(target, COLOURS - 1))
    _, placed = jax.vmap(tasks.place_progress)(
        pos, present, held, site, opening, opt['held_block'], target)
    skill_done = opt['skill_done'] | (
        opt['valid'] & jnp.where(opt['pick'], picked, placed))
    tower = (state['k'] == state['n']) & (opening > scene.OPEN_WIDTH)
    limit = jnp.where(opt['valid'], SKILL_STEPS, INVALID_STEPS)
    end = ((skill_done & ~tower) | ((steps >= limit) & ~skill_done) |
           new_obs['is_last'])
    running = ran & ~end
    opt = dict(opt, steps=steps, reward=reward, skill_done=skill_done,
               running=running)
    return state, opt, obs, running.any()

  def _select(self, mask, new, old):
    """`new` for the worlds in `mask`, `old` for the rest."""
    nd, od = new['data'], old['data']
    m = mask[:, None]

    def freeze():
      return self.base.sim.forward(nd.replace(
          qpos=jnp.where(m, nd.qpos, od.qpos),
          qvel=jnp.where(m, nd.qvel, od.qvel),
          ctrl=jnp.where(m, nd.ctrl, od.ctrl)))

    d = jax.lax.cond(mask.all(), lambda: nd, freeze)
    rest = {k: _pick(mask, new[k], old[k])
            for k in new if k not in ('data', 'key')}
    return dict(rest, data=d, key=new['key'])


class StackOptions(stack.Stack):

  @property
  def env(self):
    if self._env is None:
      self._env = BatchStackOptions(self.batch, self.size, seed=self.seed,
                                    **self.options)
    return self._env

  @property
  def obs_space(self):
    return obs_space(self.size)

  @property
  def act_space(self):
    return act_space()
