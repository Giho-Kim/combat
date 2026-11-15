#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Usage: bash scripts/train_belief_exploration.sh [agent_transitions] [new_output_directory]
CONFIG=configs/belief_exploration.json exec bash scripts/train_belief_target_allocation.sh \
  "${1:-10000000}" "${2:-runs/belief_exploration}"
