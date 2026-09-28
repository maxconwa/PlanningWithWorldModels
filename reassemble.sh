#!/usr/bin/env bash
# Rebuilds checkpoints that are stored as agent.pkl.part-* chunks, since GitHub
# LFS rejects files over 2 GiB. Run after `git lfs pull`.
set -euo pipefail
cd "$(dirname "$0")"

find . -name 'agent.pkl.part-00' -not -path './.git/*' | while read -r first; do
  dir=$(dirname "$first")
  echo "Reassembling $dir/agent.pkl"
  cat "$dir"/agent.pkl.part-* > "$dir/agent.pkl"
  (cd "$dir" && sha256sum -c agent.pkl.sha256)
done
