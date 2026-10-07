#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_target_coma_variable.sh [basic|model_based] [steps] [out]
variant="${1:-basic}"
steps="${2:-10000000}"
out="${3:-runs/target_coma_v15_variable_aligned_${variant}}"
extra=()
case "$variant" in
  basic) ;;
  model_based) extra+=(--model-based-advantage) ;;
  *) echo "Expected basic or model_based" >&2; exit 2 ;;
esac
if [[ -e "$out/config.json" ]]; then
  echo "Existing experiment: $out. Choose a new output directory." >&2
  exit 1
fi
exec "${PYTHON_BIN:-python}" -m pointmass_rl train \
  --device "${DEVICE:-auto}" --mode known --algorithm target_coma --config configs/v15_variable.json \
  --target-coma-critic graph --target-coma-actor attention \
  --steps "$steps" --out "$out" --seed "${SEED:-7}" \
  --gamma 1.0 --gae-lambda 0.97 --n-envs 32 --rollout-steps 200 \
  --epochs 10 --critic-epochs 40 --minibatch-size 256 --critic-minibatch-size 256 \
  --coma-advantage q --actor-samples all --decision-interval 1 \
  --eval-interval 100000 --eval-episodes 30 --eval-seed 10000 "${extra[@]}"
