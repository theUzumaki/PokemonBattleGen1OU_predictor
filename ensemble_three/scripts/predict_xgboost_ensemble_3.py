#!/usr/bin/env python3
"""Helper script to emit predictions using `xgboost_ensemble_3.joblib`.

Usage:
  python3 predict_xgboost_ensemble_3.py <input.jsonl> [output.csv]

    The script loads the single model `models/xgboost_ensemble_3.joblib`, the
    feature scaler `models/feature_scaler.joblib` and `models/feature_names.joblib`.
    It extracts features using package `ensemble_three.feature_extractor` and
    writes a CSV with `battle_id` and `player_won` (0/1) columns.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ensemble_three.feature_extractor import extract_all_features


def main(argv):
    if len(argv) < 2:
        print("Usage: python3 predict_xgboost_ensemble_3.py <input.jsonl> [output.csv]")
        return 2

    input_path = Path(argv[1])
    output_path = Path(argv[2]) if len(argv) > 2 else Path('ensemble_three/predictions/xgboost_ensemble_3_predictions.csv')

    MODEL_DIR = Path(__file__).parent.parent / 'models'
    MODEL_FILE = MODEL_DIR / 'xgboost_ensemble_3.joblib'
    SCALER_FILE = MODEL_DIR / 'feature_scaler.joblib'
    FEATURE_NAMES_FILE = MODEL_DIR / 'feature_names.joblib'

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 2

    if not MODEL_FILE.exists():
        print(f"Model file not found: {MODEL_FILE}")
        return 2

    if not SCALER_FILE.exists() or not FEATURE_NAMES_FILE.exists():
        print(f"Required artifacts missing in {MODEL_DIR} (scaler/feature names)")
        return 2

    # Load artifacts
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_names = joblib.load(FEATURE_NAMES_FILE)

    battles = []
    ids = []
    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            b = json.loads(line)
            battles.append(b)
            ids.append(b.get('battle_id', i))

    if len(battles) == 0:
        print("No battles found in input file.")
        return 0

    # Extract features
    features_list = [extract_all_features(b) for b in battles]
    df = pd.DataFrame(features_list)

    # Ensure feature columns exist and in right order
    for fname in feature_names:
        if fname not in df.columns:
            df[fname] = 0
    df = df[feature_names]
    df = df.fillna(0).replace([np.inf, -np.inf], 0)

    X = scaler.transform(df)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs > 0.5).astype(int)

    out_df = pd.DataFrame({'battle_id': ids, 'player_won': preds})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Saved predictions to {output_path} (rows={len(out_df)})")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
