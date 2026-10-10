# Skills

The options of the block-stacking experiment: `pick(colour)` and
`place_on(colour | table)`, each a policy on the raw 5-D end-effector action.
The environment ends a skill on the step its success condition holds, or
after 150 control steps. Starts are where other skills leave off: pick
with the arm at home or raised over the blocks, as a place leaves it; place
with the block from just clear of what is under it, as a pick leaves it, up
to 22 cm. Placing on the table puts the block on the nearest free spot.

    python skills/evaluate.py --skill pick --who scripted   # teacher gate
    python skills/student.py --skill pick --iterations 30   # image student, DAgger
    python skills/evaluate.py --skill pick --who student    # gate: >= 90% of 500
    python skills/teacher.py --skill pick                   # PPO teacher (follow-up)

- `env.py`: batched Brax environments on the stage 1 simulator; each world
  restarts on its own with a new start, so PPO runs with `wrap_env=False`.
- `tasks.py`: starts, success, and the shaped potentials PPO teachers learn
  from.
- `scripted.py`: scripted controllers from privileged state. They are the
  teachers the students learn from (pick 100%, place 98.2% of 500 starts).
- `student.py`: the image students (a CNN over both cameras, the arm, the
  target and the time fraction), distilled by DAgger; `--teacher ppo` uses a
  PPO teacher instead of the scripted one.
- `teacher.py`: PPO state teachers (Brax). The best pick teacher reaches
  73.6%; improving them is a follow-up.
- `evaluate.py`, `policy.py`: the gates, and loading skills as JAX functions
  (`load_student` for the option environment of stage 4).

## Running a student elsewhere (stage 4)

    from skills.env import student_proprio
    from skills.policy import student_params
    from skills.student import act
    params = student_params('outputs/train/skill_pick_student')   # on device
    obs = {'workspace': ..., 'wrist': ...,          # uint8 (B, 64, 64, 3)
           'proprio': student_proprio(arm, target, steps)}       # (B, 25)
    action = jax.jit(act)(params, obs)               # (B, 5) in [-1, 1]

`arm` is `environments.stack.arm_features` (19); `target` is a colour index
0-3, or 4 for the table; `steps` counts control steps since the skill
started (the time fraction is steps / 150). Passing `params` as an argument
keeps them out of the compiled step and works under DreamerV3's transfer
guard.

Runs are in `outputs/train/skill_<skill>_{teacher,student}/`; each
`config.json` records `control_steps`, the control steps spent training it.
