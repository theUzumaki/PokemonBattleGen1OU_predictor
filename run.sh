#!/usr/bin/env bash
set -euo pipefail

# Simple runner for feature extraction and correlation plotting.
# Usage:
#   ./run.sh extract    # run feature extraction
#   ./run.sh analyze    # run analysis / correlation plotting
#   ./run.sh all        # do extract then analyze
#   ./run.sh train      # run training
#   ./run.sh predict    # emit predictions for logistic and xgboost
#   ./run.sh predict_logistic       # emit predictions for logistic
#   ./run.sh predict_xgboost        # emit predictions for xgboost

# Defaults (override with environment variables)
INPUT=${INPUT:-data/train.jsonl}
FEATURES_DIR=${FEATURES_DIR:-artifacts/features}
ANALYSIS_OUT=${ANALYSIS_OUT:-artifacts/feature_analysis}
MODELS_OUT=${MODELS_OUT:-artifacts/models}
MAX_RECORDS=${MAX_RECORDS:-}

EXTRACT_CMD=(python3 -m src.feature_extractor --input "$INPUT" --output "$FEATURES_DIR")
if [ -n "$MAX_RECORDS" ]; then
  EXTRACT_CMD+=(--max-records "$MAX_RECORDS")
fi
ANALYZE_CMD=(python3 -m src.feature_analysis --features "$FEATURES_DIR/features.csv" --out "$ANALYSIS_OUT" --plot-top-k 12 --models-dir "$MODELS_OUT")

# Helper: ensure required model files exist; if missing, run training (and extract features if needed)
ensure_models() {
  local models=("$@")
  local need_train=false
  for m in "${models[@]}"; do
    if [ ! -f "$MODELS_OUT/${m}.joblib" ]; then
      need_train=true
      break
    fi
  done
  if [ "$need_train" = true ]; then
    echo "One or more required models missing: ${models[*]}. Running training..."
    FEAT_CSV="$FEATURES_DIR/features.csv"
    if [ ! -f "$FEAT_CSV" ]; then
      echo "Features CSV not found at $FEAT_CSV — extracting features first..."
      mkdir -p "$FEATURES_DIR"
      eval "${EXTRACT_CMD[*]}"
    fi
    mkdir -p "$MODELS_OUT"
    python3 -m src.train --features "$FEAT_CSV" --out "$MODELS_OUT"
  fi
}

case "${1:-help}" in
  extract)
    echo "Running feature extraction -> $FEATURES_DIR/features.csv"
    mkdir -p "$FEATURES_DIR"
    eval "${EXTRACT_CMD[*]}"
    ;;
  analyze)
    FEAT_CSV="$FEATURES_DIR/features.csv"
    if [ ! -f "$FEAT_CSV" ]; then
      echo "Features CSV not found: $FEAT_CSV"
      echo "Run './run.sh extract' first or set FEATURES_DIR to existing features output."
      exit 1
    fi
    echo "Running analysis on $FEAT_CSV -> outputs in $ANALYSIS_OUT"
    mkdir -p "$ANALYSIS_OUT"
    eval "${ANALYZE_CMD[*]}"
    ;;
  all)
    echo "Extracting then analyzing (may be slow)..."
    mkdir -p "$FEATURES_DIR" "$ANALYSIS_OUT"
    eval "${EXTRACT_CMD[*]}"
    eval "${ANALYZE_CMD[*]}"
    ;;
  full)
    echo "Full pipeline: extract -> train -> analyze"
    mkdir -p "$FEATURES_DIR" "$ANALYSIS_OUT" "$MODELS_OUT"
    echo "1/3: extracting features..."
    eval "${EXTRACT_CMD[*]}"
    echo "2/3: training models..."
    python3 -m src.train --features "$FEATURES_DIR/features.csv" --out "$MODELS_OUT"
    echo "3/3: analyzing features (with model-informed most-important feature)..."
    eval "${ANALYZE_CMD[*]}"
    ;;
  train)
    FEAT_CSV="$FEATURES_DIR/features.csv"
    if [ ! -f "$FEAT_CSV" ]; then
      echo "Features CSV not found: $FEAT_CSV"
      echo "Run './run.sh extract' first or set FEATURES_DIR to existing features output."
      exit 1
    fi
    echo "Running training on $FEAT_CSV -> outputs in $MODELS_OUT"
    mkdir -p "$MODELS_OUT"
    # call the new trainer script
    python3 -m src.train --features "$FEAT_CSV" --out "$MODELS_OUT"
    ;;
  predict)
    echo "Emitting predictions for logistic and xgboost models"
    # Use the repository test set; override by setting PREDICT_INPUT env var
    PREDICT_INPUT=${PREDICT_INPUT:-data/test.jsonl}
    # Ensure models exist (train if missing)
    ensure_models logistic xgboost
    python3 -m src.predict -m logistic -i "$PREDICT_INPUT"
    python3 -m src.predict -m xgboost -i "$PREDICT_INPUT"
    ;;
  predict_logistic)
    echo "Emitting predictions for logistic model"
    PREDICT_INPUT=${PREDICT_INPUT:-data/test.jsonl}
    ensure_models logistic
    python3 -m src.predict -m logistic -i "$PREDICT_INPUT"
    ;;
  predict_xgboost)
    echo "Emitting predictions for xgboost model"
    PREDICT_INPUT=${PREDICT_INPUT:-data/test.jsonl}
    ensure_models xgboost
    python3 -m src.predict -m xgboost -i "$PREDICT_INPUT"
    ;;
  help|--help|-h)
  echo "Usage: $0 {extract|analyze|all|train|predict|predict_logistic|predict_xgboost|help}"
  echo "Environment variables: INPUT, FEATURES_DIR, ANALYSIS_OUT, MODELS_OUT, MAX_RECORDS"
    ;;
  *)
    echo "Unknown command: ${1:-}" >&2
    echo "Usage: $0 {extract|analyze|all|train|help}"
    exit 2
    ;;
esac

exit 0
