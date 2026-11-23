#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" -m pointmass_rl.assignment_learning --config configs/target_allocation_variable.json --out "${1:-runs/target_allocation_fixed_assignment_variable}" --update-schedule interleaved --episodes 4096 --critic-updates 5000 --actor-updates 4000 --rounds 50 --log-every 10 --eval-every 10 --eval-episodes 30 --eval-seed 10000 --search-starts 64 --search-sweeps 2 --seed 7 "${@:2}"
