"""Fixes for the pinned DreamerV3 that this repository needs, applied on
import.

Agent.report draws an open-loop video for each image observation, but after
the first one it overwrites T, the report length, with the padded video's
length. With two or more image keys (custom_stack has two cameras) the next
video's border then has the wrong length and report fails. Importing this
module rebuilds Agent.report with its own name for the padded length; the
pinned source is checked first, so a change upstream fails loudly here.
"""

import inspect
import linecache
import textwrap

from dreamerv3 import agent

EDITS = (
    ('B, T, H, W, C = video.shape', 'B, VT, H, W, C = video.shape'),
    ('reshape((T, H, B * W, C))', 'reshape((VT, H, B * W, C))'),
)
FILENAME = '<dreamer_fixes: Agent.report>'


def _fix_report():
  source = textwrap.dedent(inspect.getsource(agent.Agent.report))
  for old, new in EDITS:
    assert source.count(old) == 1, f'DreamerV3 changed: {old!r}'
    source = source.replace(old, new)
  # Registered so inspect and tracebacks show the fixed source.
  linecache.cache[FILENAME] = (
      len(source), None, source.splitlines(True), FILENAME)
  namespace = {}
  exec(compile(source, FILENAME, 'exec'), vars(agent), namespace)
  report = namespace['report']
  report.__qualname__ = agent.Agent.report.__qualname__
  agent.Agent.report = report


if agent.Agent.report.__code__.co_filename != FILENAME:
  _fix_report()
