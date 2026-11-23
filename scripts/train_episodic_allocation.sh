#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" -m pointmass_rl.assignment_learning --config configs/target_allocation_variable.json --out "${1:-runs/target_allocation_episodic_variable}" "${@:2}"
