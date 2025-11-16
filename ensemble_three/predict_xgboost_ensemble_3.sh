#!/usr/bin/env bash
set -euo pipefail

# Usage: ./predict_xgboost_ensemble_3.sh <input.jsonl> [output.csv]
# Defaults: input -> data/test.jsonl, output -> ensemble_three/predictions/xgboost_ensemble_3_predictions.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root is parent of the ensemble_three directory
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Defaults: input -> <repo_root>/data/test.jsonl, output -> ensemble_three/predictions/...
INPUT_FILE="${1:-${REPO_ROOT}/data/test.jsonl}"
OUTPUT_FILE="${2:-${SCRIPT_DIR}/predictions/xgboost_ensemble_3_predictions.csv}"

# Ensure predictions directory exists
mkdir -p "${SCRIPT_DIR}/predictions"

# Run as a module so package imports (e.g. "ensemble_three") resolve correctly
# Add repo root to PYTHONPATH in case user runs this script from elsewhere
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

python3 -m ensemble_three.scripts.predict_xgboost_ensemble_3 "$INPUT_FILE" "$OUTPUT_FILE"
