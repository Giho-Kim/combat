#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

steps="${1:-10000000}"
out="${2:-runs/train_10m}"

exec python -m pointmass_rl train \
  --config configs/default.json \
  --steps "$steps" \
  --rollout-steps 256 \
  --seed 7 \
  --eval-interval 100000 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --out "$out"
