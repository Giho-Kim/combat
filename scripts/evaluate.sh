#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

model="${1:-runs/target_allocation_basic/best.pt}"
episodes="${2:-30}"
out="${3:-runs/eval_target_allocation}"

exec python -m pointmass_rl evaluate \
  --mode known --config "${CONFIG:-configs/target_allocation_variable.json}" \
  --model "$model" \
  --episodes "$episodes" \
  --seed 10000 \
  --out "$out"
