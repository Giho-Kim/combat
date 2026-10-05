#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_belief_coma.sh [agent_transitions] [new_output_directory]
exec "${PYTHON_BIN:-python}" -m pointmass_rl train \
  --mode belief --algorithm target_coma \
  --config "${CONFIG:-configs/belief.json}" \
  --steps "${1:-1000000}" --out "${2:-runs/target_coma_belief_1m}" \
  --seed "${SEED:-7}" --n-envs "${N_ENVS:-8}" \
  --rollout-steps "${ROLLOUT_STEPS:-64}" \
  --epochs "${EPOCHS:-4}" --critic-epochs "${CRITIC_EPOCHS:-4}" \
  --minibatch-size "${MINIBATCH_SIZE:-32}" \
  --eval-interval "${EVAL_INTERVAL:-100000}" \
  --eval-episodes "${EVAL_EPISODES:-30}" --eval-seed 10000
