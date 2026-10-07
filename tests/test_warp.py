"""The MJX Warp backend steps and renders the Panda on this GPU."""

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import bvh, render, render_util
from mujoco.mjx.warp import io as warp_io

from conftest import REPO

PANDA = REPO / 'environments' / 'assets' / 'panda' / 'mjx_panda.xml'


def looking_at_the_arm():
  """Quaternion for a camera at x = 1.5 looking back at the arm."""
  x, y = np.array([0, 1, 0]), np.array([-0.5, 0, 0.866])
  mat = np.stack([x, y, np.cross(x, y)], 1)
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, mat.flatten())
  return quat


def test_panda_steps_and_renders_on_gpu():
  assert jax.default_backend() == 'gpu'
  spec = mujoco.MjSpec.from_file(str(PANDA))
  spec.worldbody.add_camera(name='view', pos=[1.5, 0, 0.8],
                            quat=looking_at_the_arm())
  m = spec.compile()
  mx = mjx.put_model(m, impl='warp')
  dx = jax.vmap(lambda _: mjx.make_data(m, impl='warp'))(jnp.arange(2))
  dx = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))(mx, dx)
  ctx = warp_io.create_render_context(
      mjm=m, nworld=2, cam_res=(64, 64), use_textures=True,
      use_shadows=False, render_rgb=True, render_depth=True,
      enabled_geom_groups=[0, 1, 2])

  @jax.jit
  def draw(dx):
    dx = jax.vmap(bvh.refit_bvh, in_axes=(None, 0, None))(mx, dx, ctx.pytree())
    rgb, _, dx = jax.vmap(render.render, in_axes=(None, 0, None))(
        mx, dx, ctx.pytree())
    return jax.vmap(render_util.get_rgb, in_axes=(None, None, 0))(
        ctx.pytree(), 0, rgb)

  image = np.asarray(jax.device_get(draw(dx)))
  assert image.shape == (2, 64, 64, 3)
  assert np.isfinite(image).all() and image.std() > 0.02
