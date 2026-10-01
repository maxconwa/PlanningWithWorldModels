"""Atari Tetris with placement actions instead of joystick actions.

There are COLUMNS + 1 actions. Action c < COLUMNS moves the falling piece so
its leftmost cell is in column c, then drops it until the next piece appears.
Action COLUMNS rotates the piece once and leaves it where it is. A column the
piece cannot reach, past a wall or blocked by the stack, is clamped to the
nearest one it can, and a rotation the game refuses for lack of room does
nothing.

Every other option is DreamerV3's Atari environment, read from the atari
section of its config (--env.atari.gray False and so on). Each action runs
the joystick in closed loop against the game's RAM, so it lands where it
should with sticky actions too; the reward is the game score gained during it.

RAM layout used here (Tetris 2600):
  0x5c-0x5f  row of each of the falling piece's four cells, 0 at the bottom
  0x60-0x63  column bit of each cell in the left half: column j is bit 5 - j
  0x64-0x67  column bit of each cell in the right half: column 6 + k is bit k
  0x6b       piece shape and rotation; rotating adds 4 to bits 2-3
  0x6e       pieces spawned so far, which changes when a piece locks
"""

import ale_py
import elements
import numpy as np
from embodied.envs import atari

NOOP, FIRE, RIGHT, LEFT, DOWN = (
    ale_py.Action.NOOP, ale_py.Action.FIRE, ale_py.Action.RIGHT,
    ale_py.Action.LEFT, ale_py.Action.DOWN)

# The game reads the joystick every 4 frames, so a button held this long
# without effect is blocked, and a piece this long at its target has settled.
BLOCKED = 12
SETTLED = 8
# Frames to wait for a new piece's cells to appear in RAM.
SPAWN = 30


class Tetris(atari.Atari):

  OPTIONS = 'atari'
  COLUMNS = 10

  def __init__(self, task, actions='needed', **kwargs):
    del task, actions
    super().__init__('tetris', **kwargs)
    self.placed = None

  @property
  def act_space(self):
    space = super().act_space
    space['action'] = elements.Space(np.int32, (), 0, self.COLUMNS + 1)
    return space

  def step(self, action):
    if action['reset'] or self.done:
      return super().step({'reset': True})
    act = int(action['action'])
    assert 0 <= act <= self.COLUMNS, act
    self.reward, self.over = 0.0, False
    if self._spawned():
      self.pieces = self.ale.getRAM()[0x6e]
      if act == self.COLUMNS:
        self._rotate()
      else:
        self._place(act)
    last = self.over or self.duration >= self.length
    self.done = last
    return self._obs(self.reward, is_last=last, is_terminal=self.over)

  def _place(self, target):
    """Steer the piece's leftmost cell to `target`, drop it, and wait until the
    next piece is on screen, so the score for any lines it clears counts to
    this action and the observation shows the piece to place next."""
    best, waited = None, 0
    while not self._locked():
      cells = self._cells()
      if len(cells) < 4:
        button = DOWN
      else:
        column = min(c for _, c in cells)
        self.placed = cells
        if best is None or abs(column - target) < abs(best - target):
          best, waited = column, 0
        waited += 1
        if column != target and waited > BLOCKED:
          target = column
        button = DOWN if column == target else (
            RIGHT if column < target else LEFT)
      if not self._frame(button):
        return
    # The screen shows a new piece one frame after it appears in RAM.
    if self._spawned():
      self._frame(NOOP)

  def _rotate(self):
    """Press FIRE until the rotation advances by one, then let it settle."""
    shape = int(self.ale.getRAM()[0x6b])
    target = (shape & ~0b1100) | ((shape + 4) & 0b1100)
    pressed = settled = 0
    while not self._locked() and settled < SETTLED:
      shape = int(self.ale.getRAM()[0x6b])
      if shape == target or pressed >= BLOCKED:
        button, settled = NOOP, settled + 1
      else:
        button, pressed = FIRE, pressed + 1
      if not self._frame(button):
        return

  def _spawned(self):
    """Wait until the falling piece shows up in RAM; False if the game ended."""
    for _ in range(SPAWN):
      if len(self._cells()) == 4:
        return True
      if not self._frame(NOOP):
        return False
    return not self.over

  def _locked(self):
    return self.ale.getRAM()[0x6e] != self.pieces

  def _frame(self, button):
    """Advance the emulator one frame; False once the episode has to end."""
    self.reward += self.ale.act(button)
    self.duration += 1
    self._render()
    self.over = self.over or self.ale.game_over()
    return not (self.over or self.duration >= self.length)

  def _cells(self):
    """(row, column) of each falling cell that the game has drawn."""
    ram = self.ale.getRAM()
    cells = []
    for i in range(4):
      left, right = int(ram[0x60 + i]), int(ram[0x64 + i])
      if left:
        cells.append((int(ram[0x5c + i]), 5 - (left.bit_length() - 1)))
      elif right:
        cells.append((int(ram[0x5c + i]), 6 + right.bit_length() - 1))
    return cells
