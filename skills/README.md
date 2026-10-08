# Skills

The options of the block-stacking experiment: `pick(colour)` and
`place_on(colour | table)`, each a policy on the raw 5-D end-effector action.
The environment ends a skill on the step its success condition holds, or
after 150 control steps.

    python skills/evaluate.py --skill pick --who scripted   # teacher gate
    python skills/student.py --skill pick --iterations 30   # image student, DAgger
    python skills/evaluate.py --skill pick --who student    # gate: >= 90% of 500
    python skills/teacher.py --skill pick                   # PPO teacher (follow-up)

- `env.py`: batched Brax environments on the stage 1 simulator; each world
  restarts on its own with a new start, so PPO runs with `wrap_env=False`.
- `tasks.py`: starts, success, and the shaped potentials PPO teachers learn
  from.
- `scripted.py`: scripted controllers from privileged state. They are the
  teachers the students learn from (pick 100%, place 96.6% of 500 starts).
- `student.py`: the image students (a CNN over both cameras, the arm, the
  target and the time fraction), distilled by DAgger; `--teacher ppo` uses a
  PPO teacher instead of the scripted one.
- `teacher.py`: PPO state teachers (Brax). The best pick teacher reaches
  73.6%; improving them is a follow-up.
- `evaluate.py`, `policy.py`: the gates, and loading skills as JAX functions
  (`load_student` for the option environment of stage 4).

Runs are in `outputs/train/skill_<skill>_{teacher,student}/`; each
`config.json` records `control_steps`, the control steps spent training it.
