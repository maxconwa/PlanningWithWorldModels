# Planning

Planning inside a trained DreamerV3 world model: the agent's RSSM is the
simulator, and search runs over imagined futures instead of the real game.

- `world_model.py`: loads a run's checkpoint as a model to plan in (`build`),
  and plays the real emulator with the policy to reach a state to plan from
  (`warmup`). Also resolves run names and output paths.
- `search.py`: tree search in the world model with mctx (`make_search`).
  Nodes are expanded with the RSSM prior and edges are scored by the reward
  and continue heads.
  - `gumbel`: Gumbel MuZero with Sequential Halving at the root; leaves are
    valued by the critic.
  - `puct`: MuZero's MCTS with the policy as prior; leaves are valued by the
    critic.
  - `uct`: UCB1 with no policy prior; leaves are valued by policy rollouts in
    the world model.
- `eval_model.py`: plays full games on the real emulator, acting with the
  policy or a planner, and reports scores.
- `search_tree.py`: draws one search tree, with the frame the decoder
  predicts at every node.
- `dream_atari.py`: an interactive window for playing inside the world
  model's imagination.

Run the scripts from the repository root:

    python planning/eval_model.py --logdir tetris_400m_latest --planner gumbel --sims 32
    python planning/search_tree.py --logdir tetris_400m_latest --out tree.html

`--logdir` takes a run name in `outputs/train/` or a path. Results go to
`outputs/eval/<run>/`.
