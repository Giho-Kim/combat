#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-runs/target_allocation_episodic_ppo_variable}"
if [[ -e "$out/config.json" ]]; then
  echo "Existing experiment: $out. Choose a new output directory." >&2
  exit 1
fi
exec "${PYTHON_BIN:-python}" -m pointmass_rl train --mode known --algorithm target_allocation --config configs/target_allocation_variable.json --episodic-assignments --assignment-episodes 256 --steps 10000000 --out "$out" --seed 7 --gamma 1.0 --gae-lambda 0.97 --n-envs 32 --epochs 10 --critic-epochs 20 --minibatch-size 256 --critic-minibatch-size 256 --eval-interval 100000 --eval-episodes 30 --eval-seed 10000 "${@:2}"
