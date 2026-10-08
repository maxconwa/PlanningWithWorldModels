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


def student_params(path):
  """A trained student's parameters, on the device. Pass them into jitted
  code with skills.student.act(params, obs), so they are an argument rather
  than constants baked into every compiled step."""
  params = pickle.loads((pathlib.Path(path) / 'params.pkl').read_bytes())
  return jax.device_put(params)


def load_student(path):
  """The image student's policy: obs dict with workspace, wrist (uint8
  (B, 64, 64, 3)) and proprio (skills.env.student_proprio) to actions
  (B, 5)."""
  from skills import student
  params = student_params(path)
  return lambda obs: student.act(params, obs)
