import inspect

import dreamer_fixes  # noqa: F401
from dreamerv3 import agent


def test_report_keeps_its_report_length_across_cameras():
  """Each camera's open-loop video uses its own padded length, not T."""
  source = inspect.getsource(agent.Agent.report)
  assert 'B, T, H, W, C = video.shape' not in source
  assert 'B, VT, H, W, C = video.shape' in source
  assert agent.Agent.report.__globals__ is vars(agent)
