#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Reproduce the successful PPO fine-tuning stage from its frozen initialization.
# Usage: PYTHON_BIN=... bash scripts/train_target_coma_model_free.sh [new_output] [initial.pt]
# This is a weight warm-start with fresh optimizers, not training from scratch.
out="${1:-runs/target_coma_model_free_retrain}"
initial="${2:-runs/tune_v13_commit_batch128_s43/initial.pt}"

exec "${PYTHON_BIN:-python}" -u -m scripts.tune_target_coma \
  --variant mc_commit \
  --actor-distances \
  --epochs 10 \
  --critic-epochs 4 \
  --n-envs 128 \
  --learning-rate .001 \
  --entropy-coef .005 \
  --b12-probability .5 \
  --steps 1000000 \
  --seed 43 \
  --eval-interval 100000 \
  --init-model "$initial" \
  --out "$out"
