#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_belief_target_allocation.sh [agent_transitions] [new_output_directory]
exec "${PYTHON_BIN:-python}" -m pointmass_rl train \
  --mode belief --algorithm target_allocation \
  --config "${CONFIG:-configs/belief.json}" \
  --steps "${1:-1000000}" --out "${2:-runs/belief_target_allocation_1m}" \
  --seed "${SEED:-7}" --n-envs "${N_ENVS:-8}" \
  --rollout-steps "${ROLLOUT_STEPS:-64}" \
  --epochs "${EPOCHS:-4}" --critic-epochs "${CRITIC_EPOCHS:-4}" \
  --minibatch-size "${MINIBATCH_SIZE:-32}" \
  --learning-rate "${LEARNING_RATE:-5e-4}" --entropy-coef "${ENTROPY_COEF:-0.01}" \
  --eval-interval "${EVAL_INTERVAL:-100000}" \
  --eval-episodes "${EVAL_EPISODES:-30}" --eval-seed 10000
