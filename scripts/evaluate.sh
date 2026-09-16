#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model="${1:-runs/train_10m/best.pt}"
episodes="${2:-100}"
out="${3:-runs/eval_10m}"

exec python -m pointmass_rl evaluate \
  --config configs/default.json \
  --model "$model" \
  --episodes "$episodes" \
  --seed 10000 \
  --out "$out"
