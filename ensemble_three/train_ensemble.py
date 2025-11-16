"""
Train an XGBoost bagging ensemble (simple, regularized, low-overfit focus).

This script trains `n_models` XGBoost classifiers on bootstrap samples
of the training set, using subsampling/colsample and regularization.
Each model is saved to `models/xgboost_ensemble_{i}.joblib` and an
`ensemble_meta.json` describing the ensemble is written to `models/`.

Usage: run from `ensemble_three/` directory.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List

from .train import load_battle_data, extract_features_and_labels, preprocess_features
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
import argparse


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def train_bagging_ensemble(n_models: int = 5, random_seed: int = RANDOM_SEED, data_path: str = '../data/train.jsonl'):
    print("Loading data and extracting features...")
    # Data path anchored to repository: ../data relative to ensemble_three package
    battles = load_battle_data(BASE_DIR.parent / 'data' / 'train.jsonl')
    X_df, y = extract_features_and_labels(battles)

    # Split into train/test for evaluation and early stopping
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    # Preprocess (fit scaler on train, transform test)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_df, X_test_df)

    model_paths: List[str] = []

    # Ensemble training params (regularized to avoid overfitting)
    base_params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'random_state': None,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }

    for i in range(n_models):
        seed = random_seed + i
        print(f"\nTraining ensemble member {i+1}/{n_models} (seed={seed})...")

        # Bootstrap sample indices from training set
        rng = np.random.RandomState(seed)
        n_train = X_train_scaled.shape[0]
        indices = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train_scaled[indices]
        y_boot = y_train[indices]

        params = dict(base_params)
        params['random_state'] = seed

        model = XGBClassifier(**params)

        # Use early stopping on the holdout test set to avoid overfitting
        # Fit without early stopping (regularized + subsampling used to reduce overfitting)
        model.fit(X_boot, y_boot, verbose=False)

        model_path = MODEL_DIR / f'xgboost_ensemble_{i+1}.joblib'
        joblib.dump(model, model_path)
        model_paths.append(str(model_path.name))
        print(f"  ✓ Saved model to {model_path}")

    # Save scaler and feature names
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"✓ Saved scaler to {scaler_path}")

    feature_names_path = MODEL_DIR / 'feature_names.joblib'
    joblib.dump(list(X_df.columns), feature_names_path)
    print(f"✓ Saved feature names to {feature_names_path}")

    # Ensemble metadata
    meta = {
        'type': 'xgboost_bagging_ensemble',
        'n_models': n_models,
        'models': model_paths,
        'random_seed': random_seed
    }

    meta_path = MODEL_DIR / 'ensemble_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Ensemble metadata saved to {meta_path}")
    print("Ensemble training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost bagging ensemble.")
    parser.add_argument('--data_path', type=str, default='../data/train.jsonl',
                        help='Path to training data JSONL file')
    parser.add_argument('--n_models', type=int, default=5,
                        help='Number of ensemble models to train')
    args = parser.parse_args()

    # Pass the data path to train_bagging_ensemble
    train_bagging_ensemble(n_models=args.n_models, random_seed=RANDOM_SEED, data_path=args.data_path)


if __name__ == '__main__':
    main()
