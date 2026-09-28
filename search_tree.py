"""Draw the search tree the agent builds in its world model.

Plays the real emulator with the trained policy for --step frames, then runs
one search (mctx) from the posterior RSSM state of the last frame: Gumbel
MuZero with Sequential Halving at the root (--planner gumbel), or MCTS with
PUCT (--planner puct). Graphviz lays the tree out top to bottom. Every node, the root included,
shows the frame the decoder produces for its RSSM state, so the whole tree is
drawn by the world model. Each edge is the action taken from its parent.

  ~/anaconda3/envs/dreamer/bin/python search_tree.py --logdir tetris_400m_latest
  ~/anaconda3/envs/dreamer/bin/python search_tree.py --logdir tetris_400m_latest \
      --step 350 --sims 64 --out tree_350.svg
  ~/anaconda3/envs/dreamer/bin/python search_tree.py --logdir tetris_400m_latest \
      --planner puct --out puct_tree.html

The --out extension picks the output: .html is an interactive page (click a
node for its statistics), .svg a standalone vector file with the frames
embedded, and any other Graphviz format (.png, .pdf) is rendered by dot
directly. Graphviz lives in its own conda env; --dot points at another binary.
"""

import argparse
import base64
import io
import json
import math
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

from common import PUCT, build, logdir_arg, make_search, seeder, warmup

import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
from PIL import Image

DOT = shutil.which('dot') or str(
    pathlib.Path.home() / 'anaconda3/envs/graphviz/bin/dot')
FONT = 'DejaVu Sans Mono'
FRAME = 96  # Points per frame side; one point per frame pixel in the SVG.

# Graph colors, written into the dot file as the light theme. The HTML page
# maps each one back to its theme token so the graph follows dark mode too.
COLORS = dict(
    surface='#f8f9fc', ink='#1b2030', muted='#5e6578', line='#c6cad6',
    accent='#6446d4', good='#1d7f47', bad='#bf3d28')


def make_decoder(model):
  """Jitted decoder from a batch of RSSM states to uint8 frames."""

  def decode(carry):
    _, _, recons = model.dec(
        {}, carry, jnp.zeros(len(carry['deter']), bool), False, single=True)
    return jnp.clip(255 * recons['image'].pred(), 0, 255).astype(jnp.uint8)

  return jax.jit(nj.pure(decode))


def halving_rounds(considered, sims):
  """The Sequential Halving schedule mctx follows, as [actions, sims] rounds."""
  if considered <= 1:
    return [[1, sims]]
  log2max = math.ceil(math.log2(considered))
  rounds, used, num = [], 0, considered
  while used < sims:
    take = min(max(1, int(sims / (log2max * num))) * num, sims - used)
    if rounds and rounds[-1][0] == num:
      rounds[-1][1] += take
    else:
      rounds.append([num, take])
    used += take
    num = max(2, num // 2)
  return rounds


def png(image):
  image = image[..., 0] if image.shape[-1] == 1 else image
  buf = io.BytesIO()
  Image.fromarray(image).save(buf, format='PNG')
  return buf.getvalue()


def data_uri(data):
  return 'data:image/png;base64,' + base64.b64encode(data).decode()


def tree_data(parts, out, frames, real, run, step, sims, planner, considered):
  """Everything the outputs draw, as plain JSON-ready values."""
  tree = out.search_tree
  t = jax.device_get(dict(
      visits=tree.node_visits, value=tree.node_values, raw=tree.raw_values,
      parents=tree.parents, action=tree.action_from_parent,
      children=tree.children_index, logits=tree.children_prior_logits,
      rewards=tree.children_rewards, discounts=tree.children_discounts,
      cvisits=tree.children_visits, cvalues=tree.children_values,
      weights=out.action_weights, chosen=out.action))
  horizon = parts['model'].config.horizon
  actions = parts['actions']
  softmax = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
  r4 = lambda x: float(f'{float(x):.5g}')

  # The chosen action, then the most visited child at each level below it.
  pv, node = {0}, int(t['children'][0, t['chosen']])
  while node >= 0 and t['visits'][node] > 0:
    pv.add(node)
    kids = t['children'][node]
    best = int(np.argmax(np.where(kids >= 0, t['cvisits'][node], -1)))
    node = int(kids[best]) if t['cvisits'][node, best] > 0 else -1

  nodes, depth = [], {0: 0}
  for n in np.flatnonzero(t['visits'] > 0):
    n, parent = int(n), int(t['parents'][n])
    item = dict(id=n, parent=parent, visits=int(t['visits'][n]),
                value=r4(t['value'][n]), raw=r4(t['raw'][n]), pv=n in pv,
                png=png(frames[n]))
    if parent >= 0:
      a = int(t['action'][n])
      depth[n] = depth[parent] + 1
      reward, discount = t['rewards'][parent, a], t['discounts'][parent, a]
      item.update(
          action=a, reward=r4(reward),
          cont=r4(min(1.0, discount / (1 - 1 / horizon))),
          prior=r4(softmax(t['logits'][parent])[a]),
          q=r4(reward + discount * t['value'][n]))
    item['depth'] = depth[n]
    nodes.append(item)

  prior, n = softmax(t['logits'][0]), t['cvisits'][0]
  q = t['rewards'][0] + t['discounts'][0] * t['cvalues'][0]
  # action_weights: the improved policy for Gumbel, the visit shares for PUCT.
  root_actions = [dict(
      action=a, prior=r4(prior[a]), visits=int(n[a]),
      q=r4(q[a]) if n[a] > 0 else None, improved=r4(t['weights'][a]))
      for a in range(len(actions))]

  extra = {}
  if planner == 'gumbel':
    gumbel = jax.device_get(tree.extra_data.root_gumbel)
    for r in root_actions:
      r['gumbel'] = r4(gumbel[r['action']])
    # The actions in each round are the ones with the most simulations; the
    # improved policy breaks ties the way the final pick does.
    order = sorted(range(len(actions)), key=lambda a: (-n[a], -t['weights'][a]))
    extra['rounds'] = [dict(size=size, sims=total, actions=order[:size])
                       for size, total in halving_rounds(considered, sims)]
  else:
    # The PUCT scores at the root once the search is done, computed the way
    # mctx does: Q min-max scaled against the root value and the visited
    # siblings (unvisited actions get 0), plus the exploration bonus U.
    v, root_n = t['value'][0], t['visits'][0]
    safe = np.where(n > 0, q, v)
    lo, hi = min(v, safe.min()), max(v, safe.max())
    qhat = (np.where(n > 0, q, lo) - lo) / max(hi - lo, 1e-8)
    c = PUCT['pb_c_init'] + np.log(
        (root_n + PUCT['pb_c_base'] + 1) / PUCT['pb_c_base'])
    u = np.sqrt(root_n) * c * prior / (n + 1)
    for r in root_actions:
      r.update(qhat=r4(qhat[r['action']]), u=r4(u[r['action']]))
    extra['puct'] = dict(c=r4(c), c1=PUCT['pb_c_init'], c2=PUCT['pb_c_base'])

  task = parts['task']
  return dict(
      game=task.split('_', 1)[1].replace('_', ' ').title(), task=task,
      run=run, ckpt=parts['ckpt'], step=step, sims=sims, planner=planner,
      considered=considered, actions=actions, chosen=int(t['chosen']),
      policyTop=int(np.argmax(prior)), rootActions=root_actions,
      nodes=nodes, real=png(real), **extra)


def to_dot(data, paths):
  """Graphviz source for the tree, reading each node's frame from `paths`."""
  C, A = COLORS, data['actions']
  lines = [
      'digraph search {',
      f'  graph [rankdir=TB, ordering=out, nodesep=0.2, ranksep=0.3, '
      f'pad=0.25, bgcolor="{C["surface"]}"];',
      f'  node [shape=plain, fontname="{FONT}", fontsize=10, '
      f'fontcolor="{C["muted"]}"];',
      f'  edge [fontname="{FONT}", fontsize=10, color="{C["line"]}", '
      f'arrowsize=0.6];',
  ]
  kids = {}
  for n in data['nodes']:
    root, end = n['parent'] < 0, n['parent'] >= 0 and n['cont'] < 0.5
    border = (C['ink'] if root else C['bad'] if end
              else C['accent'] if n['pv'] else C['line'])
    style = ' STYLE="dashed"' if end else ''
    label = (
        '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="0">'
        f'<TR><TD BORDER="2" COLOR="{border}"{style} FIXEDSIZE="TRUE" '
        f'WIDTH="{FRAME}" HEIGHT="{FRAME}">'
        f'<IMG SRC="{paths[n["id"]]}" SCALE="BOTH"/></TD></TR>'
        f'<TR><TD CELLPADDING="3">N {n["visits"]} · V {n["value"]:.2f}</TD></TR>'
        '</TABLE>>')
    cls = ' '.join(c for c, on in (('root', root), ('pv', n['pv']), ('end', end)) if on)
    lines.append(f'  n{n["id"]} [id="n{n["id"]}", class="{cls}", label={label}];')
    kids.setdefault(n['parent'], []).append(n)

  # With ordering=out, siblings appear in edge order: most searched on the left.
  for parent, children in kids.items():
    if parent < 0:
      continue
    for k in sorted(children, key=lambda k: (-k['visits'], k['action'])):
      color = C['accent'] if k['pv'] else C['line']
      label = (f'<FONT COLOR="{C["accent"] if k["pv"] else C["ink"]}">'
               f'{A[k["action"]]}</FONT><BR/>'
               f'<FONT POINT-SIZE="9" COLOR="{C["muted"]}">N {k["visits"]}</FONT>')
      if k['reward'] >= 0.05:
        label += (f'<FONT POINT-SIZE="9" COLOR="{C["good"]}">'
                  f' r +{k["reward"]:.1f}</FONT>')
      width = 1 + 5 * math.sqrt(k['visits'] / data['sims'])
      lines.append(
          f'  n{parent} -> n{k["id"]} [id="e{k["id"]}", '
          f'class="{"pv" if k["pv"] else "side"}", penwidth={width:.2f}, '
          f'color="{color}", label=<{label}>];')
  lines.append('}')
  return '\n'.join(lines)


def run_dot(source, fmt, binary):
  if not pathlib.Path(binary).exists():
    sys.exit(f'Graphviz dot not found at {binary}. Install it with\n'
             '  ~/anaconda3/bin/conda create -n graphviz -c conda-forge graphviz\n'
             'or pass --dot /path/to/dot.')
  result = subprocess.run([binary, f'-T{fmt}'], input=source.encode(),
                          capture_output=True)
  if result.returncode:
    sys.exit(f'dot failed:\n{result.stderr.decode()}')
  return result.stdout


def embed(svg, paths, data):
  """Standalone SVG: frames inlined as data URIs, Graphviz comments dropped."""
  svg = svg[svg.index('<svg'):]
  svg = re.sub(r'<!--.*?-->\n?', '', svg, flags=re.S)
  svg = re.sub(r'<title>.*?</title>\n?', '', svg, flags=re.S)
  pngs = {n['id']: n['png'] for n in data['nodes']}
  for n, path in paths.items():
    svg = svg.replace(f'xlink:href="{path}"', f'xlink:href="{data_uri(pngs[n])}"')
  return svg


def page(data, svg):
  graph_colors = '\n'.join(
      f'.graph [fill="{hex_}"] {{ fill: var(--{name}); }}\n'
      f'.graph [stroke="{hex_}"] {{ stroke: var(--{name}); }}'
      for name, hex_ in COLORS.items())
  info = dict(data, real=data_uri(data['real']), nodes=[
      dict({k: v for k, v in n.items() if k != 'png'}, img=data_uri(n['png']))
      for n in data['nodes']])
  title = f'{data["game"]} {"Search" if data["planner"] == "gumbel" else "PUCT"} Tree'
  return HEAD + (PAGE.replace('__TITLE__', title)
                 .replace('__GRAPH_COLORS__', graph_colors)
                 .replace('__SVG__', svg)
                 .replace('__DATA__', json.dumps(info).replace('</', '<\\/')))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', type=logdir_arg, default='tetris_400m_latest',
                      help='run directory, or a run name next to this script')
  parser.add_argument('--step', type=int, default=200,
                      help='real frames played with the policy before the search')
  parser.add_argument('--planner', choices=['gumbel', 'puct'], default='gumbel',
                      help='Gumbel MuZero search, or MCTS with PUCT')
  parser.add_argument('--sims', type=int, default=32)
  parser.add_argument('--considered', type=int, default=16,
                      help='root actions Sequential Halving starts from '
                           '(--planner gumbel; capped at the action count)')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--out', default='search_tree.html',
                      help='.html page, .svg, or any other Graphviz format')
  parser.add_argument('--dot', default=DOT, help='Graphviz dot binary')
  args = parser.parse_args()
  fmt = pathlib.Path(args.out).suffix.lstrip('.').lower() or 'html'
  _, seed = seeder(args.seed)

  print('Loading the agent...')
  parts = build(args.logdir, env_seed=args.seed)
  considered = min(args.considered, len(parts['actions']))
  print(f'Playing {args.step} real frames with the policy...')
  dyn_carry, obs = warmup(parts, args.step, seed)

  if args.planner == 'gumbel':
    print(f'Searching with Gumbel MuZero: {args.sims} simulations over '
          f'{considered} actions...')
    batched = make_search(parts['model'], args.sims, 'gumbel',
                          max_num_considered_actions=considered)
  else:
    print(f'Searching with MCTS (PUCT): {args.sims} simulations...')
    batched = make_search(parts['model'], args.sims, 'puct', **PUCT)
  # Drop the batch axis inside jit; indexing device arrays outside of it
  # would trip DreamerV3's host-to-device transfer guard.
  search = jax.jit(lambda *a: jax.tree.map(lambda x: x[0], batched(*a)))
  out = search(parts['params'], dyn_carry, seed())
  _, frames = make_decoder(parts['model'])(
      parts['params'], out.search_tree.embeddings, seed=seed())
  data = tree_data(parts, out, jax.device_get(frames), obs['image'],
                   args.logdir.name, args.step, args.sims, args.planner,
                   considered)

  with tempfile.TemporaryDirectory() as tmp:
    paths = {}
    for n in data['nodes']:
      paths[n['id']] = f'{tmp}/n{n["id"]}.png'
      pathlib.Path(paths[n['id']]).write_bytes(n['png'])
    source = to_dot(data, paths)
    if fmt in ('html', 'svg'):
      svg = embed(run_dot(source, 'svg', args.dot).decode(), paths, data)
      text = page(data, svg) if fmt == 'html' else svg
      pathlib.Path(args.out).write_text(text)
    else:
      pathlib.Path(args.out).write_bytes(run_dot(source, fmt, args.dot))
  chosen = data['actions'][data['chosen']]
  print(f'Search plays {chosen} from {len(data["nodes"])} nodes. '
        f'Wrote {args.out}')


# A full document when opened from disk. The lines after HEAD are also a
# valid page body on their own.
HEAD = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""

PAGE = """<title>__TITLE__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Silkscreen&display=swap">
<style>
/* Layout: summary of the root decision on top, then the Graphviz tree running
   top to bottom in its own scroll pane beside a sticky node inspector. */
:root {
  --bg: #eceef3;
  --surface: #f8f9fc;
  --ink: #1b2030;
  --muted: #5e6578;
  --line: #c6cad6;
  --accent: #6446d4;
  --accent-soft: #e3ddfa;
  --good: #1d7f47;
  --bad: #bf3d28;
  --screen: #0c0e14;
  --display: 'Silkscreen', 'Courier New', monospace;
  --body: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --data: 'IBM Plex Mono', 'DejaVu Sans Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #11141b; --surface: #191d28; --ink: #e2e5ee; --muted: #959cae;
    --line: #3a4152; --accent: #a792ff; --accent-soft: #2b2548;
    --good: #4fc583; --bad: #f27d66; --screen: #05060a; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #11141b; --surface: #191d28; --ink: #e2e5ee; --muted: #959cae;
  --line: #3a4152; --accent: #a792ff; --accent-soft: #2b2548;
  --good: #4fc583; --bad: #f27d66; --screen: #05060a; color-scheme: dark;
}
[hidden] { display: none !important; }
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.5 var(--body);
}
.page {
  display: grid; gap: 20px; max-width: 1480px; margin: 0 auto;
  padding-inline: 16px; padding-block: 24px 40px;
}
h1, h2, h3 { margin: 0; text-wrap: balance; }
h1 { font: 400 clamp(22px, 4vw, 32px)/1.1 var(--display); letter-spacing: 0.02em; }
h2 { font-size: 15px; font-weight: 600; }
h3 { font-size: 13px; font-weight: 600; color: var(--muted); }
.eyebrow {
  margin: 0; font: 500 11px/1.4 var(--data); text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--muted);
}
.head { display: grid; gap: 8px; }
.verdict { margin: 0; max-width: 65ch; }
.verdict b { color: var(--accent); font-family: var(--data); font-weight: 500; }
td, .stats dd, .kid { font-variant-numeric: tabular-nums; }

.summary {
  display: grid; gap: 16px;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1.4fr);
}
.panel {
  background: var(--surface); border: 1px solid var(--line); border-radius: 6px;
  padding: 16px; display: grid; gap: 12px; align-content: start; min-width: 0;
}
.rule { display: grid; gap: 12px; }
.rule-text { margin: 0; font-size: 13px; color: var(--muted); max-width: 62ch; }
.rounds { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.shares { list-style: none; margin: 0; padding: 0; display: grid; gap: 6px; }
.shares li {
  display: grid; grid-template-columns: 5.5em minmax(0, 1fr) 4em; gap: 10px;
  align-items: center; font: 500 12px/1.4 var(--data);
}
.shares .track { height: 10px; background: var(--bg); border: 1px solid var(--line); border-radius: 2px; overflow: hidden; }
.shares .fill { display: block; height: 100%; background: var(--line); }
.shares li.pick .fill { background: var(--accent); }
.shares li.pick .name { color: var(--accent); }
.shares .count { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; }
.round { display: grid; grid-template-columns: 6.5em minmax(0, 1fr); gap: 4px 12px; }
.round .label { font: 500 12px/1.6 var(--data); color: var(--muted); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip {
  font: 500 12px/1 var(--data); padding: 5px 8px; border-radius: 3px;
  border: 1px solid var(--line); background: var(--bg); color: var(--ink);
}
.chip.out { color: var(--muted); text-decoration: line-through; }
.chip.pick { background: var(--accent); border-color: var(--accent); color: var(--surface); }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th {
  text-align: right; font: 500 11px/1.4 var(--data); text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--muted); padding: 0 8px 6px;
}
td { text-align: right; padding: 5px 8px; border-top: 1px solid var(--line); font-family: var(--data); }
th:first-child, td:first-child { text-align: left; }
tr.is-chosen td { background: var(--accent-soft); }
tr.is-chosen td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
.bar { display: inline-block; height: 8px; background: var(--accent); vertical-align: middle; margin-left: 6px; border-radius: 1px; }

.workspace {
  display: grid; gap: 16px; align-items: start;
  grid-template-columns: minmax(0, 1fr) 340px;
}
.canvas-panel { padding: 0; gap: 0; }
.toolbar {
  display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center;
  padding: 10px 16px; border-bottom: 1px solid var(--line);
}
.toolbar label { font-size: 13px; color: var(--muted); display: flex; gap: 8px; align-items: center; }
.toolbar output { font: 400 12px var(--data); min-width: 4ch; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12px; color: var(--muted); margin: 0; padding: 0; list-style: none; }
.legend span { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -2px; margin-right: 6px; }
.key-root { border: 2px solid var(--ink); }
.key-pv { border: 2px solid var(--accent); }
.key-end { border: 2px dashed var(--bad); }
.key-rew { background: var(--good); }
.tree-scroll { overflow: auto; max-height: 80vh; padding: 8px; }
.tree-scroll svg { display: block; margin-inline: auto; }

/* The Graphviz drawing, recoloured from its light palette to theme tokens. */
__GRAPH_COLORS__
.graph text { font-family: var(--data); }
.graph image { image-rendering: pixelated; }
.graph g.node { cursor: pointer; outline: none; }
.graph g.node:hover polygon { stroke: var(--accent); }
.graph g.node.selected polygon { stroke: var(--accent); stroke-width: 5px; }
.graph g.node:focus-visible polygon { stroke: var(--ink); stroke-width: 5px; }

.inspector { position: sticky; top: calc(env(safe-area-inset-top, 0px) + 16px); }
.frames { display: flex; gap: 10px; flex-wrap: wrap; }
.frames figure { margin: 0; display: grid; gap: 4px; flex: 1 1 140px; max-width: 300px; }
.frames img { width: 100%; height: auto; image-rendering: pixelated; background: var(--screen); border-radius: 3px; display: block; }
.frames figcaption { font-size: 12px; color: var(--muted); }
.stats { display: grid; grid-template-columns: auto auto; gap: 4px 16px; margin: 0; font-size: 13px; }
.stats dt { color: var(--muted); }
.stats dd { margin: 0; text-align: right; font-family: var(--data); }
.stats dd.good { color: var(--good); }
.stats dd.bad { color: var(--bad); }
.kids { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
.kid {
  width: 100%; display: grid; grid-template-columns: 1fr auto auto; gap: 12px;
  font: 400 12px/1.4 var(--data); color: var(--ink); background: var(--bg);
  border: 1px solid var(--line); border-radius: 3px; padding: 5px 8px; cursor: pointer; text-align: left;
}
.kid:hover { border-color: var(--accent); }
.kid:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.note { margin: 0; font-size: 12px; color: var(--muted); max-width: 65ch; }

@media (max-width: 980px) {
  .summary, .workspace { grid-template-columns: minmax(0, 1fr); }
  .inspector { position: static; order: -1; }
}
</style>

<main class="page">
  <header class="head">
    <p class="eyebrow" id="meta"></p>
    <h1>__TITLE__</h1>
    <p class="verdict" id="verdict"></p>
  </header>

  <section class="summary">
    <div class="panel">
      <h2 id="rule-title"></h2>
      <div class="rule" id="rule"></div>
    </div>
    <div class="panel">
      <h2>Root actions</h2>
      <div class="table-wrap"><table id="root-table"></table></div>
    </div>
  </section>

  <section class="workspace">
    <div class="panel canvas-panel">
      <div class="toolbar">
        <label for="zoom">Zoom
          <input type="range" id="zoom" min="40" max="200" step="10" value="100">
          <output for="zoom" id="zoom-value">100%</output>
        </label>
        <ul class="legend">
          <li><span class="key-root"></span>Root</li>
          <li><span class="key-pv"></span>Chosen line</li>
          <li><span class="key-end"></span>Game over predicted</li>
          <li><span class="key-rew"></span>Line clear predicted</li>
        </ul>
      </div>
      <div class="tree-scroll">__SVG__</div>
    </div>
    <aside class="panel inspector" id="inspector" aria-live="polite"></aside>
  </section>

  <p class="note">Every frame in the tree is drawn by the world model's decoder,
  including the root, which decodes the RSSM state after the real frame. Nodes
  below it are states the search imagined, sampled once per node. Each edge is
  the action taken from its parent, with the simulations that passed through it
  (N) and any predicted reward (r). V is the mean value the search backed up
  through a node.</p>
</main>

<script type="application/json" id="tree-data">__DATA__</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('tree-data').textContent);
  const A = D.actions;
  const fmt = (x, d = 3) => x === null || x === undefined ? '–' : x.toFixed(d);
  // Probabilities: the policy is often near-certain, so keep tiny ones visible.
  const prob = p => p >= 0.001 ? p.toFixed(3) : p > 0 ? p.toExponential(0) : '0';
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  const index = new Map(D.nodes.map(n => [n.id, Object.assign(n, { kids: [] })]));
  for (const n of D.nodes) if (n.parent >= 0) index.get(n.parent).kids.push(n);
  for (const n of D.nodes) n.kids.sort((a, b) => b.visits - a.visits || a.action - b.action);
  const root = index.get(0);
  const path = n => {
    const out = [];
    for (let m = n; m.parent >= 0; m = index.get(m.parent)) out.unshift(A[m.action]);
    return out;
  };

  // Header and root summary.
  const gumbel = D.planner === 'gumbel';
  document.getElementById('meta').textContent =
    `${D.run} · checkpoint ${D.ckpt} · real frame ${D.step} · `
    + `${gumbel ? 'Gumbel MuZero' : 'MCTS with PUCT'} · ${D.sims} simulations`;
  const top = D.rootActions[D.policyTop];
  const verdict = document.getElementById('verdict');
  verdict.append(gumbel ? 'Search picks ' : 'MCTS plays ', el('b', '', A[D.chosen]),
    gumbel ? '. ' : ', its most visited root action. ');
  verdict.append(D.chosen === D.policyTop
    ? `The policy alone would most likely have picked the same action (π ${prob(top.prior)}).`
    : `The policy alone would most likely have picked `);
  if (D.chosen !== D.policyTop)
    verdict.append(el('b', '', A[D.policyTop]), ` (π ${prob(top.prior)}).`);

  const rule = document.getElementById('rule');
  document.getElementById('rule-title').textContent =
    gumbel ? 'Sequential Halving at the root' : 'PUCT at the root';
  const rounds = el('ol', 'rounds');
  if (gumbel) D.rounds.forEach((r, i) => {
    const li = el('li', 'round');
    const each = r.sims / r.size;
    li.append(el('span', 'label', `Round ${i + 1}`));
    const chips = el('div', 'chips');
    const next = D.rounds[i + 1];
    for (const a of r.actions) {
      const out = next ? !next.actions.includes(a) : a !== D.chosen;
      chips.append(el('span', 'chip' + (out ? ' out' : ''), A[a]));
    }
    li.append(chips, el('span'), el('span', 'label',
      `${r.size} actions × ${Number.isInteger(each) ? each : '≈' + Math.round(each)} simulations`));
    rounds.append(li);
  });
  const byVisits = (a, b) => b.visits - a.visits || b.improved - a.improved;
  if (gumbel) {
    const pick = el('li', 'round');
    const pickChips = el('div', 'chips');
    pickChips.append(el('span', 'chip pick', A[D.chosen]));
    pick.append(el('span', 'label', 'Pick'), pickChips);
    rounds.append(pick);
    rule.append(rounds);
  } else {
    rule.append(el('p', 'rule-text', 'Every simulation walks down from the root '
      + 'taking the action with the largest Q̂ + U. Q̂ is Q scaled to [0, 1] among '
      + 'siblings, and U = c · P · √N / (1 + n) favours actions the policy likes '
      + 'but the search has tried little. The most visited root action is played.'));
    const shares = el('ul', 'shares');
    for (const r of [...D.rootActions].sort(byVisits)) {
      const li = el('li', r.action === D.chosen ? 'pick' : '');
      const track = el('span', 'track'), fill = el('span', 'fill');
      fill.style.width = `${(100 * r.visits / D.sims).toFixed(1)}%`;
      track.append(fill);
      li.append(el('span', 'name', A[r.action]), track, el('span', 'count', `N ${r.visits}`));
      shares.append(li);
    }
    rule.append(shares, el('p', 'rule-text',
      `c = ${fmt(D.puct.c)} at the root after ${root.visits} visits `
      + `(c₁ ${D.puct.c1}, c₂ ${D.puct.c2}).`));
  }

  // Root table columns: [heading, cell] per planner.
  const withBar = (text, share) => {
    const td = el('td', '', text);
    const bar = el('span', 'bar');
    bar.style.width = `${Math.round(share * 48)}px`;
    td.append(bar);
    return td;
  };
  const cell = text => el('td', '', text);
  let columns, rows;
  if (gumbel) {
    // The final pick is the finalist with the highest g + log π′.
    const finalists = D.rounds[D.rounds.length - 1].actions;
    const score = r => finalists.includes(r.action) ? r.gumbel + Math.log(r.improved) : null;
    rows = [...D.rootActions].sort((a, b) => b.visits - a.visits
      || (score(b) ?? -Infinity) - (score(a) ?? -Infinity) || b.improved - a.improved);
    columns = [
      ['Action', r => cell(A[r.action])], ['Prior π', r => cell(prob(r.prior))],
      ['Gumbel g', r => cell(fmt(r.gumbel, 2))], ['N', r => cell(String(r.visits))],
      ['Q', r => cell(fmt(r.q))], ['After search π′', r => withBar(prob(r.improved), r.improved)],
      ['g + log π′', r => cell(fmt(score(r), 2))]];
  } else {
    rows = [...D.rootActions].sort(byVisits);
    columns = [
      ['Action', r => cell(A[r.action])], ['Prior P', r => cell(prob(r.prior))],
      ['N', r => cell(String(r.visits))], ['Q', r => cell(fmt(r.q))],
      ['Q̂', r => cell(fmt(r.qhat))], ['U', r => cell(fmt(r.u))],
      ['Q̂ + U', r => cell(fmt(r.qhat + r.u))],
      ['Visit share', r => withBar(fmt(r.improved, 2), r.improved)]];
  }
  const table = document.getElementById('root-table');
  const head = el('tr');
  for (const [h] of columns) head.append(el('th', '', h));
  table.append(head);
  for (const r of rows) {
    const tr = el('tr', r.action === D.chosen ? 'is-chosen' : '');
    for (const [, render] of columns) tr.append(render(r));
    table.append(tr);
  }

  // The Graphviz tree: make its nodes selectable and let the zoom resize it.
  const scroller = document.querySelector('.tree-scroll');
  const svg = scroller.querySelector('svg');
  const box = svg.viewBox.baseVal;
  svg.setAttribute('aria-label', `Search tree with ${D.nodes.length} nodes`);
  const groups = new Map();
  for (const n of D.nodes) {
    const g = svg.querySelector(`#n${n.id}`);
    if (!g) continue;
    g.setAttribute('tabindex', '0');
    g.setAttribute('role', 'button');
    g.setAttribute('aria-label', n.parent < 0 ? 'Root of the search'
      : `${path(n).join(', ')}: ${n.visits} simulations`);
    g.addEventListener('click', () => select(n));
    g.addEventListener('keydown', e => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(n); }
    });
    groups.set(n.id, g);
  }

  // Scroll the tree pane so a node is in view, or centred when asked.
  function reveal(n, centre) {
    const g = groups.get(n.id);
    if (!g) return;
    const r = g.getBoundingClientRect(), s = scroller.getBoundingClientRect();
    const x = r.left - s.left + scroller.scrollLeft + r.width / 2;
    const y = r.top - s.top + scroller.scrollTop + r.height / 2;
    const inView = r.left >= s.left && r.right <= s.right && r.top >= s.top && r.bottom <= s.bottom;
    if (centre || !inView) {
      scroller.scrollLeft = x - scroller.clientWidth / 2;
      scroller.scrollTop = y - scroller.clientHeight / 2;
    }
  }

  const zoom = document.getElementById('zoom');
  const zoomValue = document.getElementById('zoom-value');
  function applyZoom() {
    const s = +zoom.value / 100;
    svg.setAttribute('width', box.width * s);
    svg.setAttribute('height', box.height * s);
    zoomValue.textContent = `${zoom.value}%`;
    reveal(selected, true);
  }

  const inspector = document.getElementById('inspector');
  let selected = root;
  function select(n) {
    groups.get(selected.id)?.classList.remove('selected');
    selected = n;
    groups.get(n.id)?.classList.add('selected');
    const isRoot = n.parent < 0;
    const parts = [el('p', 'eyebrow', isRoot ? 'Root · after the real frame'
      : `Depth ${n.depth} · imagined${n.pv ? ' · chosen line' : ''}`)];
    parts.push(el('h2', '', isRoot ? `Real frame ${D.step}` : path(n).join(' → ')));

    const frames = el('div', 'frames');
    const figure = (src, text) => {
      const f = el('figure');
      const img = el('img');
      img.src = src;
      img.alt = text;
      f.append(img, el('figcaption', '', text));
      return f;
    };
    frames.append(figure(n.img, 'World model frame'));
    if (isRoot) frames.append(figure(D.real, 'Real frame, for reference'));
    parts.push(frames);

    const stats = el('dl', 'stats');
    const stat = (k, v, cls) => { stats.append(el('dt', '', k), el('dd', cls || '', v)); };
    stat('Simulations through node', String(n.visits));
    stat('Search value V', fmt(n.value));
    stat('Value head at expansion', fmt(n.raw));
    if (!isRoot) {
      stat('Prior π of this action', prob(n.prior));
      stat('Predicted reward', fmt(n.reward), n.reward >= 0.05 ? 'good' : '');
      stat('Continue probability', fmt(n.cont, 2), n.cont < 0.5 ? 'bad' : '');
      stat('Q of the edge in', fmt(n.q));
    } else {
      stat('Plays', A[D.chosen]);
    }
    parts.push(stats);

    if (n.kids.length) {
      parts.push(el('h3', '', 'Children'));
      const list = el('ul', 'kids');
      for (const k of n.kids) {
        const li = el('li');
        const b = el('button', 'kid');
        b.type = 'button';
        b.append(el('span', '', A[k.action]), el('span', '', `N ${k.visits}`),
          el('span', '', `Q ${fmt(k.q)}`));
        b.addEventListener('click', () => { select(k); reveal(k); });
        li.append(b);
        list.append(li);
      }
      parts.push(list);
    }
    if (!isRoot) {
      const up = el('button', 'kid', 'Back to parent');
      up.type = 'button';
      up.style.gridTemplateColumns = '1fr';
      up.addEventListener('click', () => { const p = index.get(n.parent); select(p); reveal(p); });
      parts.push(up);
    }
    inspector.replaceChildren(...parts);
  }

  zoom.addEventListener('input', applyZoom);
  select(root);
  applyZoom();
})();
</script>
"""


if __name__ == '__main__':
  main()
