#!/usr/bin/env bash
# Drone rescue simulation. Allocation targets are sites requiring rescue assistance.
set -euo pipefail
cd "$(dirname "$0")/.."
# Each worker runs its own simulation. Avoid nested BLAS thread pools.
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 BLIS_NUM_THREADS=1
# Usage: bash scripts/train_search_final_mission_termination.sh [episodes] [new_output_directory]
# Both checkpoints stay frozen. Only the termination fusion actor and critic are trained.
exec "${PYTHON_BIN:-python}" -u -m pointmass_rl.search_final_mission \
  --exploration-checkpoint "${EXPLORATION_CHECKPOINT:-runs/belief_exploration_decision_ppo_gpu_______varying/best.pt}" \
  --allocation-checkpoint "${ALLOCATION_CHECKPOINT:-runs/target_allocation_variable_aligned_basic_warmstart/latest.pt}" \
  --config "${CONFIG:-configs/belief_exploration.json}" \
  --episodes "${1:-20000}" --out "${2:-runs/search_final_mission_termination}" \
  --seed "${SEED:-7}" --device "${DEVICE:-cpu}" --workers "${WORKERS:-16}" \
  --episodes-per-update "${EPISODES_PER_UPDATE:-32}" \
  --epochs "${EPOCHS:-4}" --critic-epochs "${CRITIC_EPOCHS:-4}" \
  --minibatch-size "${MINIBATCH_SIZE:-64}" --learning-rate "${LEARNING_RATE:-0.0003}" \
  --entropy-coef "${ENTROPY_COEF:-0.01}" --policy-clip "${POLICY_CLIP:-0.2}" \
  --allocation-horizon "${ALLOCATION_HORIZON:-100}" \
  --decision-interval "${DECISION_INTERVAL:-20}" --forced-margin "${FORCED_MARGIN:-5}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL:-5000}" \
  --eval-interval "${EVAL_INTERVAL:-640}" --eval-episodes "${EVAL_EPISODES:-30}" --eval-seed "${EVAL_SEED:-10000}"
