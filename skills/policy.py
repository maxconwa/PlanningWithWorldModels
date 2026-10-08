"""Trained skills as JAX functions, loaded from their run directories."""

import json
import pathlib
import pickle

import jax


def load_teacher(path):
  """The teacher's deterministic policy: state obs (B, 85) to actions
  (B, 5)."""
  from brax.training.acme import running_statistics
  from brax.training.agents.ppo import networks
  from skills.env import ACTION_DIM, STATE_DIM
  from skills.teacher import network_factory
  path = pathlib.Path(path)
  params = pickle.loads((path / 'params.pkl').read_bytes())
  network = network_factory()(
      {'state': (STATE_DIM,)}, ACTION_DIM,
      preprocess_observations_fn=running_statistics.normalize)
  act = networks.make_inference_fn(network)(params, deterministic=True)
  key = jax.random.PRNGKey(0)
  return lambda state: act({'state': state}, key)[0]


def config(path):
  return json.loads((pathlib.Path(path) / 'config.json').read_text())


def load_student(path):
  """The image student's policy: obs dict with workspace, wrist and
  proprio to actions (B, 5)."""
  from skills import student
  params = pickle.loads((pathlib.Path(path) / 'params.pkl').read_bytes())
  return lambda obs: student.act(params, obs)
