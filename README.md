**Project Overview**

This repository contains tools and models for predicting outcomes of Gen 1 OU Pokémon battles.
The codebase includes feature extraction, model training, and ensemble prediction utilities
used to create the shipped models and predictions in `artifacts/` and `models/`.

**Quick Start**

- **Create venv**: `python -m venv .venv` then `source .venv/bin/activate`
- **Install deps**: `pip install -r requirements.txt` (additional viz deps: `pip install pandas seaborn matplotlib`)
- **Run a quick feature-correlation check**:
   `python -m src.feature_correlation --input data/train.jsonl --output out --max-records 100`

**What the feature-correlation script produces**

- **`out/features_table.csv`**: Extracted numeric features per record
- **`out/correlation_*.csv`**: Correlation matrices (Pearson, Spearman)
- **`out/correlation_*_heatmap.png`**: Heatmap visualizations (if enabled)
- **`out/top_corr_*.txt`**: Top correlated feature pairs

**Repository Layout (important files)**

- **`src/`**: Primary source code (feature extraction, training, prediction, analysis)
- **`data/`**: Datasets used for training and testing (`train.jsonl`, `test.jsonl`, plus helpers)
- **`models/`**: Saved ensemble models and metadata (`xgboost_ensemble_*.joblib`, `feature_names.joblib`)
- **`artifacts/`**: Outputs from training and analysis (feature importances, scalers)
- **`ensemble_three/`**: Ensemble training and prediction utilities; contains its own `models/` and `scripts/`
- **`predictions_*.csv`**: Precomputed prediction outputs for convenience
- **`requirements.txt`**: Python dependencies
- **`run.sh`, `emit_pred.sh`**: Convenience scripts for running the pipeline or exporting predictions

**Common Tasks**

- **Extract features and compute correlations**:
   `python -m src.feature_correlation --input data/train.jsonl --output out --max-records 100`
- **Train models**:
   Check `src/train.py` and `ensemble_three/train.py` for single-model and ensemble training flows.
- **Predict with an ensemble**:
   Use scripts in `ensemble_three/scripts/` (for example, `predict_xgboost_ensemble_3.py`) or the top-level predictor utilities in `src/predict.py`.

**Helpful scripts**

- **`run.sh`**: Top-level orchestration for common pipeline steps (see script header for usage)
- **`emit_pred.sh`**: Convenience wrapper to emit predictions in a specific CSV layout
- **`ensemble_three/predict_xgboost_ensemble_3.sh`**: Example ensemble prediction script

**Notes & tips**

- The repository assumes a reasonably recent Python 3 interpreter (3.8+ recommended).
- Use `--max-records` for fast iterations on large datasets.
- Models and scalers are serialized with `joblib` in `models/` and `ensemble_three/models/`.

**Contributing / Extending**

- To add features, update `src/feature_extractor.py` and re-run training.
- To add a new model, implement training in `src/train.py` and add a predict wrapper in `src/predict.py`.

**License & Contact**

- This repository is shared for research and educational use. If you plan to reuse code or data,
   please check any attached license files or contact the maintainer.
