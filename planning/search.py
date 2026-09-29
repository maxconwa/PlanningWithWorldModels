"""Tree search inside a learned world model, with mctx.

Import world_model first: it puts DreamerV3 on the path, which make_search
needs when it builds the search.
"""

import jax
import ninjax as nj

# MCTS with PUCT as played at evaluation: MuZero's constants, no root Dirichlet
# noise (it only serves exploration in training), and the most visited root
# action is played.
PUCT = dict(pb_c_init=1.25, pb_c_base=19652.0, dirichlet_fraction=0.0,
            temperature=0.0)

# UCT: UCB1 exploration constant for Q values scaled to [0, 1], and leaf
# values from `rollouts` policy rollouts of `horizon` imagined steps.
UCT = dict(c=2 ** 0.5, rollouts=8, horizon=50, bootstrap=False)


def uct_args(parser):
  """Command line flags for --planner uct, defaulting to UCT."""
  parser.add_argument('--c', type=float, default=UCT['c'],
                      help='UCB1 exploration constant (--planner uct)')
  parser.add_argument('--rollouts', type=int, default=UCT['rollouts'],
                      help='policy rollouts averaged per leaf (--planner uct)')
  parser.add_argument('--horizon', type=int, default=UCT['horizon'],
                      help='imagined steps per rollout (--planner uct)')
  parser.add_argument('--bootstrap', action='store_true',
                      help='add the critic at the end of each rollout '
                           '(--planner uct)')


def uct_options(args):
  return {k: getattr(args, k) for k in UCT}


def make_search(model, sims, planner='gumbel', **options):
  """Tree search (mctx) inside the world model.

  Returns search(params, dyn_carry, key) -> mctx.PolicyOutput for a batch of
  posterior RSSM states. Nodes are expanded with the RSSM prior, which samples
  its stochastic state once per node, and edges are scored by the reward and
  continue heads. The planners differ in how they descend and value leaves:

    gumbel  Gumbel MuZero: the root draws actions from the policy without
            replacement and Sequential Halving splits the simulations between
            them. Leaves are valued by the critic. `options` go to
            mctx.gumbel_muzero_policy (max_num_considered_actions).
    puct    MuZero's MCTS: PUCT with the policy as prior everywhere, acting by
            visit counts. Leaves are valued by the critic. `options` go to
            mctx.muzero_policy (see PUCT).
    uct     UCT: UCB1 on scaled Q values, trying every action once first; the
            policy plays no part in the tree. Leaves are valued by the mean
            discounted return of policy rollouts in the world model, plus the
            critic at their end if `bootstrap`. Acts by visit counts. `options`
            are the keys of UCT.
  """
  import jax.numpy as jnp
  import mctx
  from dreamerv3.agent import sample

  f32 = lambda x: x.astype(jnp.float32)
  # With contdisc the continue head is trained on the discounted target, so
  # its probability is already the per-step discount.
  disc = 1.0 if model.config.contdisc else 1 - 1 / model.config.horizon

  def heads(carry):
    inp = model.feat2tensor(carry)
    return f32(model.pol(inp, 1)['action'].logits), f32(model.val(inp, 1).pred())

  def step(carry, action):
    carry, _ = model.dyn.imagine(carry, {'action': action}, 1, False, single=True)
    inp = model.feat2tensor(carry)
    logits, value = heads(carry)
    return carry, dict(
        reward=f32(model.rew(inp, 1).pred()),
        discount=disc * f32(model.con(inp, 1).prob(1)),
        prior_logits=logits, value=value)

  def rollout_step(carry):
    action = sample(model.pol(model.feat2tensor(carry), 1))
    carry, _ = model.dyn.imagine(carry, action, 1, False, single=True)
    inp = model.feat2tensor(carry)
    return carry, (f32(model.rew(inp, 1).pred()),
                   disc * f32(model.con(inp, 1).prob(1)))

  root_fn, step_fn = nj.pure(heads), nj.pure(step)
  rollout_fn = nj.pure(rollout_step)

  def rollout_value(params, carry, key):
    """Mean discounted return of policy rollouts from a batch of states."""
    k = options['rollouts']
    batch = len(carry['deter'])
    carry = jax.tree.map(lambda x: jnp.repeat(x, k, 0), carry)

    def body(state, key):
      carry, ret, weight = state
      _, (carry, (reward, discount)) = rollout_fn(params, carry, seed=key)
      return (carry, ret + weight * reward, weight * discount), None

    key, tail_key = jax.random.split(key)
    ones = jnp.ones(batch * k, jnp.float32)
    (carry, ret, weight), _ = jax.lax.scan(
        body, (carry, 0 * ones, ones), jax.random.split(key, options['horizon']))
    if options['bootstrap']:
      _, (_, tail) = root_fn(params, carry, seed=tail_key)
      ret = ret + weight * tail
    return ret.reshape(batch, k).mean(1)

  def recurrent_fn(params, key, action, carry):
    key, rollout_key = jax.random.split(key)
    _, (carry, out) = step_fn(params, carry, action, seed=key)
    if planner == 'uct':
      out['value'] = rollout_value(params, carry, rollout_key)
    return mctx.RecurrentFnOutput(**out), carry

  def ucb1(key, tree, node):
    """Untried actions first, in random order, then the largest
    Q̂ + c·sqrt(ln N / n) with Q̂ min-max scaled among siblings."""
    visits = tree.children_visits[node]
    qhat = mctx.qtransform_by_parent_and_siblings(tree, node)
    bonus = options['c'] * jnp.sqrt(
        jnp.log(tree.node_visits[node]) / jnp.maximum(visits, 1))
    noise = jax.random.uniform(key, visits.shape)
    score = jnp.where(visits > 0, qhat + bonus + 1e-7 * noise, 1e6 + noise)
    return jnp.argmax(score).astype(jnp.int32)

  def search(params, dyn_carry, key):
    key, root_key, rollout_key = jax.random.split(key, 3)
    _, (logits, value) = root_fn(params, dyn_carry, seed=root_key)
    if planner == 'uct':
      value = rollout_value(params, dyn_carry, rollout_key)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value, embedding=dyn_carry)
    if planner != 'uct':
      policy = dict(gumbel=mctx.gumbel_muzero_policy,
                    puct=mctx.muzero_policy)[planner]
      return policy(params, key, root, recurrent_fn, num_simulations=sims,
                    **options)
    tree = mctx.search(
        params, key, root=root, recurrent_fn=recurrent_fn,
        root_action_selection_fn=ucb1,
        interior_action_selection_fn=lambda key, tree, node, depth: ucb1(
            key, tree, node),
        num_simulations=sims)
    # Most visited root action, ties broken by the scaled Q.
    visits = tree.children_visits[:, 0]
    qhat = jax.vmap(mctx.qtransform_by_parent_and_siblings, (0, None))(
        tree, jnp.int32(0))
    return mctx.PolicyOutput(
        action=jnp.argmax(visits + 0.5 * qhat, -1).astype(jnp.int32),
        action_weights=visits / visits.sum(-1, keepdims=True),
        search_tree=tree)

  return search
