"""Lets Brax 0.14.2 run on JAX 0.10.2.

Brax's PPO calls jax.device_put_replicated (brax/training/pmap.py), which
JAX 0.10 no longer exports; the implementation is still in jax._src.api.
Importing this module restores the name; on a JAX that still has it, it does
nothing. skills/__init__.py imports it.
"""

import jax
from jax._src import api

try:
  jax.device_put_replicated
except AttributeError:
  jax.device_put_replicated = api.device_put_replicated
