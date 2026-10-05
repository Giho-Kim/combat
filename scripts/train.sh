#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

steps="${1:-1000000}"
out="${2:-runs/train_1m}"

exec python -m pointmass_rl train \
  --config configs/default.json \
  --steps "$steps" \
  --rollout-steps 200 \
  --n-envs 32 \
  --decision-interval "${DECISION_INTERVAL:-1}" \
  --seed 7 \
  --eval-interval 100000 \
  --eval-episodes 30 \
  --eval-seed 10000 \
  --out "$out"
