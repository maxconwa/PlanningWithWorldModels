"""Batched block-stacking simulation on the GPU.

MJX with the Warp backend steps every world's physics (MuJoCo Warp) and
renders both cameras (MuJoCo Warp's batch renderer) as JAX functions over a
batch of worlds on axis 0. Every stage of the experiment simulates and
renders through this module.
"""

import os

os.environ.setdefault('MJX_WARP_ENABLED', 'true')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402
from mujoco import mjx  # noqa: E402
from mujoco.mjx._src import bvh, render, render_util  # noqa: E402
from mujoco.mjx.warp import io as warp_io  # noqa: E402

from environments import stack_scene as scene  # noqa: E402

# Contacts and constraint rows each world has room for. MuJoCo Warp's
# defaults drop contacts once four blocks rest on the floor; these cover
# every block on the floor or in a stack plus both finger pads.
CONTACTS_PER_WORLD = 64
# Warp captures each MJX call as a CUDA graph. Its default (WARP) fails with
# "capturing stream has unjoined work" inside DreamerV3's training loop;
# WARP_STAGED does not, and is as fast (NONE is four times slower).
GRAPH_MODE = wp.JaxCallableGraphMode.WARP_STAGED
NJMAX = 384


class Sim:

  def __init__(self, num_envs, size=(64, 64)):
    self.num_envs, self.size = num_envs, tuple(size)
    self.mjm = m = scene.build_model()
    self.model = mjx.put_model(m, impl='warp', graph_mode=GRAPH_MODE)
    self.site = m.site('gripper').id
    self.bodies = np.array(scene.block_body_ids(m))
    self.qpos_adr = scene.block_qpos_adr(m)
    self.dof_adr = scene.block_dof_adr(m)
    self.cameras = (m.camera('workspace').id, m.camera('wrist').id)
    self.home_qpos = jax.device_put(scene.home_qpos(m).astype(np.float32))
    self.home_ctrl = jax.device_put(scene.home_ctrl().astype(np.float32))
    self.context = warp_io.create_render_context(
        mjm=m, nworld=num_envs, cam_res=(self.size[1], self.size[0]),
        use_textures=True, use_shadows=False, render_rgb=True,
        render_depth=True, enabled_geom_groups=[0, 1, 2])

  def make_data(self):
    """Every world at home, at rest."""
    m = self.mjm
    # naconmax counts contacts over the whole batch; njmax is per world.
    naconmax = CONTACTS_PER_WORLD * self.num_envs
    d = jax.vmap(lambda _: mjx.make_data(
        m, impl='warp', naconmax=naconmax, njmax=NJMAX))(
            jnp.arange(self.num_envs))
    return self.forward(self.home(d))

  def home(self, d):
    b = self.num_envs
    return d.replace(
        qpos=jnp.broadcast_to(self.home_qpos, (b,) + self.home_qpos.shape),
        qvel=jnp.zeros_like(d.qvel),
        ctrl=jnp.broadcast_to(self.home_ctrl, (b,) + self.home_ctrl.shape))

  def set_blocks(self, d, positions, yaws):
    """Every world's blocks at `positions`, turned by `yaws` about z, at
    rest."""
    quat = jnp.stack([jnp.cos(yaws / 2), jnp.zeros_like(yaws),
                      jnp.zeros_like(yaws), jnp.sin(yaws / 2)], -1)
    qpos, qvel = d.qpos, d.qvel
    for c, (a, v) in enumerate(zip(self.qpos_adr, self.dof_adr)):
      qpos = qpos.at[:, a:a + 3].set(positions[:, c])
      qpos = qpos.at[:, a + 3:a + 7].set(quat[:, c])
      qvel = qvel.at[:, v:v + 6].set(0)
    return self.forward(d.replace(qpos=qpos, qvel=qvel))

  def physics(self, d, ctrl, steps):
    step = jax.vmap(mjx.step, in_axes=(None, 0))
    d = d.replace(ctrl=ctrl)
    return jax.lax.fori_loop(0, steps, lambda _, d: step(self.model, d), d)

  def forward(self, d):
    return jax.vmap(mjx.forward, in_axes=(None, 0))(self.model, d)

  def render(self, d):
    """Both cameras for every world, as uint8 RGB."""
    ctx = self.context.pytree()
    d = jax.vmap(bvh.refit_bvh, in_axes=(None, 0, None))(self.model, d, ctx)
    rgb, _, d = jax.vmap(render.render, in_axes=(None, 0, None))(
        self.model, d, ctx)
    images = []
    for camera in self.cameras:
      image = jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
          ctx, camera, rgb)
      images.append(jnp.clip(image * 255, 0, 255).astype(jnp.uint8))
    return images[0], images[1], d

  def block_pos(self, d):
    return d.xpos[:, self.bodies]

  def block_quat(self, d):
    return d.xquat[:, self.bodies]

  def block_vel(self, d):
    return jnp.stack([d.qvel[:, v:v + 3] for v in self.dof_adr], 1)

  def site_pos(self, d):
    return d.site_xpos[:, self.site]

  def opening(self, d):
    return d.qpos[:, 7] + d.qpos[:, 8]
