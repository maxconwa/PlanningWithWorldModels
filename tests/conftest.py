"""Puts the repository and the DreamerV3 checkout on sys.path. JAX does not
preallocate GPU memory, so MuJoCo Warp can allocate beside it."""

import os
import pathlib
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MJX_WARP_ENABLED', 'true')
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / 'third_party' / 'dreamerv3')]
