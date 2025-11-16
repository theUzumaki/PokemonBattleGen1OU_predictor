#!/usr/bin/env bash
set -euo pipefail

# Usage: emit_pred.sh <logistic|xgboost|ensemble> [repo_root] [input.jsonl] [output.csv]
# If repo_root is provided (e.g. /kaggle/working/PokemonBattleGen1OU_predictor) the script
# will run commands relative to that directory. On Kaggle provide the unzipped dataset path.

MODE=${1:-}
REPO_ROOT=${2:-.}
INPUT_FILE=${3:-}
OUTPUT_FILE=${4:-}

INPUT_KAGGLE_TRAIN=${INPUT_KAGGLE_TRAIN:-}/kaggle/input/fds-pokemon-battles-prediction-2025/train.jsonl
INPUT_KAGGLE_TEST=${INPUT_KAGGLE_TEST:-}/kaggle/input/fds-pokemon-battles-prediction-2025/test.jsonl

if [ -z "$MODE" ]; then
    echo "Usage: $0 <logistic|xgboost|ensemble> [repo_root] [input.jsonl] [output.csv]"
    exit 1
fi

if [ ! -d "$REPO_ROOT" ]; then
    echo "Repository root not found: $REPO_ROOT"
    exit 2
fi

# Export PYTHONPATH so module-style invocations find package modules
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ensure_models_artifact() {
    # $1 = model key (e.g. logistic, xgboost)
    local key="$1"
    local model_file="$REPO_ROOT/artifacts/models/${key}.joblib"
    if [ -f "$model_file" ]; then
        echo "Model present: $model_file"
        return 0
    fi
    echo "Model not found: $model_file"
    # Ensure features exist
    local feat_csv="$REPO_ROOT/artifacts/features/features.csv"
    if [ ! -f "$feat_csv" ]; then
        echo "Features CSV missing: $feat_csv — extracting features"
        (cd "$REPO_ROOT" && bash run.sh extract)
    fi
    echo "Training models to produce $model_file"
    (cd "$REPO_ROOT" && bash run.sh train)
    if [ ! -f "$model_file" ]; then
        echo "Training finished but model still missing: $model_file" >&2
        return 1
    fi
    echo "Model trained: $model_file"
}

ensure_ensemble_artifact() {
    local model_file="$REPO_ROOT/ensemble_three/models/xgboost_ensemble_3.joblib"
    if [ -f "$model_file" ]; then
        echo "Ensemble model present: $model_file"
        return 0
    fi
    echo "Ensemble model not found: $model_file"
    # Try to run the ensemble training script
    if [ -f "$REPO_ROOT/ensemble_three/train_ensemble.py" ]; then
        echo "Running ensemble trainer..."
        (cd "$REPO_ROOT" && PYTHONPATH="$PYTHONPATH" python3 ensemble_three/train_ensemble.py)
    else
        echo "No ensemble trainer found at ensemble_three/train_ensemble.py; cannot train ensemble." >&2
        return 1
    fi
    if [ ! -f "$model_file" ]; then
        echo "Ensemble training finished but model still missing: $model_file" >&2
        return 1
    fi
    echo "Ensemble model trained: $model_file"
}

case "$MODE" in
    logistic)
        # Optionally pass input file via $INPUT_FILE or use default
        if [ -n "$INPUT_FILE" ]; then
            export PREDICT_INPUT="$INPUT_FILE"
        fi
        ensure_models_artifact logistic
        (cd "$REPO_ROOT" && bash run.sh predict_logistic)
        ;;
    xgboost)
        if [ -n "$INPUT_FILE" ]; then
            export PREDICT_INPUT="$INPUT_FILE"
        fi
        ensure_models_artifact xgboost
        (cd "$REPO_ROOT" && bash run.sh predict_xgboost)
        ;;
    ensemble)
        # ensemble expects input and output; default to repo test set and ensemble predictions path
        if [ -n "$INPUT_FILE" ]; then
            IN_ARG="$INPUT_FILE"
        else
            IN_ARG="$REPO_ROOT/data/test.jsonl"
        fi
        if [ -n "$OUTPUT_FILE" ]; then
            OUT_ARG="$OUTPUT_FILE"
        else
            OUT_ARG="$REPO_ROOT/ensemble_three/predictions/xgboost_ensemble_3_predictions.csv"
        fi
        ensure_ensemble_artifact
        (cd "$REPO_ROOT" && bash ensemble_three/predict_xgboost_ensemble_3.sh "$IN_ARG" "$OUT_ARG")
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Usage: $0 <logistic|xgboost|ensemble> [repo_root] [input.jsonl] [output.csv]"
        exit 3
        ;;
esac

exit 0