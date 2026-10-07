#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Usage: bash scripts/train_target_coma.sh [agent_transitions] [output_directory]
steps="${1:-1000000}"
out="${2:-runs/target_coma_v15_attention_1m}"

# Keep existing experiments/checkpoints intact.
if [[ -e "$out/config.json" ]]; then
  echo "Existing experiment: $out. Choose a new output directory." >&2
  exit 1
fi

exec "${PYTHON_BIN:-python}" -m pointmass_rl train \
  --device "${DEVICE:-auto}" --mode known \
  --algorithm target_coma \
  --target-coma-critic "${CRITIC:-graph}" \
  --target-coma-actor "${ACTOR:-attention}" \
  --config configs/five_agents.json \
  --steps "$steps" \
  --out "$out" \
  --seed "${SEED:-7}" \
  --n-envs "${N_ENVS:-32}" \
  --rollout-steps 200 \
  --epochs "${EPOCHS:-10}" \
  --critic-epochs "${CRITIC_EPOCHS:-40}" \
  --minibatch-size 256 \
  --critic-minibatch-size 256 \
  --coma-advantage q \
  --actor-samples all \
  --decision-interval "${DECISION_INTERVAL:-1}" \
  --eval-interval 100000 \
  --eval-episodes 30 \
  --eval-seed 10000
