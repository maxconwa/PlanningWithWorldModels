"""Lets the pinned DreamerV3 run on JAX versions with keyword-only jax.jit.

DreamerV3 (third_party/dreamerv3) targets JAX 0.4.33 and passes in_shardings,
out_shardings, static_argnums, static_argnames and donate_argnums to jax.jit
positionally. Newer JAX, which newer GPUs such as the RTX 50 series need,
takes them only by keyword. Importing this module wraps jax.jit to pass them
by keyword; on a JAX that still accepts them positionally it does nothing.
"""

import functools
import inspect

import jax

NAMES = ('in_shardings', 'out_shardings', 'static_argnums', 'static_argnames',
         'donate_argnums')


def _keyword_only():
  params = list(inspect.signature(jax.jit).parameters.values())
  return len(params) > 1 and params[1].kind == inspect.Parameter.KEYWORD_ONLY


if _keyword_only() and not hasattr(jax.jit, 'positional'):
  _jit = jax.jit

  @functools.wraps(_jit)
  def jit(fun=None, *args, **kwargs):
    kwargs.update(zip(NAMES, args))
    return _jit(**kwargs) if fun is None else _jit(fun, **kwargs)

  jit.positional = True
  jax.jit = jit
