"""Convert DreamerV3 metrics.jsonl logs into TensorBoard event files.

Writes one event file into each run directory that has a metrics.jsonl, so the
runs show up under their folder names. Re-running replaces the previous events.

  ~/anaconda3/envs/torch/bin/python jsonl_to_tensorboard.py ~/WorldModels
  ~/anaconda3/bin/tensorboard --logdir ~/WorldModels
"""

import argparse
import json
import math
import pathlib

from torch.utils.tensorboard import SummaryWriter


def convert(rundir):
  for old in rundir.glob('events.out.tfevents.*'):
    old.unlink()
  writer = SummaryWriter(str(rundir))
  count = 0
  with (rundir / 'metrics.jsonl').open() as f:
    for line in f:
      row = json.loads(line)
      step = int(row.pop('step'))
      for key, value in row.items():
        if isinstance(value, (int, float)) and math.isfinite(value):
          writer.add_scalar(key, value, step)
          count += 1
  writer.close()
  return count


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('root', type=pathlib.Path, nargs='?',
                      default=pathlib.Path('/home/max/WorldModels'))
  args = parser.parse_args()
  for metrics in sorted(args.root.glob('*/metrics.jsonl')):
    print(f'{metrics.parent.name}: {convert(metrics.parent)} scalars')


if __name__ == '__main__':
  main()
