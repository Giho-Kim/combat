#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Each worker runs its own simulation. Avoid nested BLAS thread pools.
export OPENBLAS_NUM_THREADS="${BLAS_THREADS:-1}"
export MKL_NUM_THREADS="${BLAS_THREADS:-1}"
export OMP_NUM_THREADS="${BLAS_THREADS:-1}"
export BLIS_NUM_THREADS="${BLAS_THREADS:-1}"
# Usage: bash scripts/train_belief_exploration.sh [agent_transitions] [new_output_directory]
# Actor inputs use absolute map positions and both previous intent channels.
# ROLLOUT_DECISIONS counts polygon selections, not physics steps. Each collection
# finishes complete episode waves with frozen weights; the budget may overshoot.
exploration_args=()
if [[ "${EVERY_STEP:-0}" == "1" ]]; then
  exploration_args+=(--exploration-every-step)
fi
exec "${PYTHON_BIN:-python}" -u -m pointmass_rl train \
  --mode belief --algorithm target_allocation --exploration-ppo "${exploration_args[@]}" \
  --device "${DEVICE:-cuda}" \
  --config "${CONFIG:-configs/belief_exploration.json}" \
  --steps "${1:-10000000}" --out "${2:-runs/belief_exploration_decision_ppo_absolute}" \
  --seed "${SEED:-7}" --n-envs "${N_ENVS:-8}" \
  --rollout-steps "${ROLLOUT_DECISIONS:-64}" \
  --epochs "${EPOCHS:-4}" --critic-epochs "${CRITIC_EPOCHS:-4}" \
  --minibatch-size "${MINIBATCH_SIZE:-32}" \
  --learning-rate "${LEARNING_RATE:-0.0005}" \
  --entropy-coef "${ENTROPY_COEF:-0.01}" --policy-clip "${POLICY_CLIP:-0.2}" \
  --exploration-checkpoint-interval "${CHECKPOINT_INTERVAL:-500000}" \
  --eval-interval "${EVAL_INTERVAL:-500000}" \
  --eval-revisit-guard \
  --eval-workers "${EVAL_WORKERS:-4}" \
  --eval-screen-episodes "${EVAL_SCREEN_EPISODES:-0}" \
  --eval-episodes "${EVAL_EPISODES:-10}" --eval-seed "${EVAL_SEED:-10000}"
