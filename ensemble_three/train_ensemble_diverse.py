"""
Train a diverse XGBoost ensemble by varying hyperparameters per member.

This script creates `n_models` where each member samples from a small
hyperparameter grid to increase diversity (and thus reduce correlated errors).
Saves models to `models/xgboost_ensemble_diverse_{i}.joblib` and metadata to
`models/ensemble_meta_diverse.json`.

Run from `ensemble_three/`.
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List

from train import load_battle_data, extract_features_and_labels, preprocess_features
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


MODEL_DIR = Path('./models')
MODEL_DIR.mkdir(exist_ok=True)
DATA_FILE = Path('../data/train.jsonl')

RANDOM_SEED = 42


def sample_params(rng: np.random.RandomState):
    # Small grid to sample from
    max_depth = int(rng.choice([3, 4, 5]))
    learning_rate = float(rng.choice([0.03, 0.05, 0.08]))
    n_estimators = int(rng.choice([100, 150, 200]))
    subsample = float(rng.choice([0.7, 0.8, 0.9]))
    colsample = float(rng.choice([0.6, 0.8, 1.0]))
    reg_alpha = float(rng.choice([0.0, 0.5, 1.0]))
    reg_lambda = float(rng.choice([1.0, 2.0]))

    return {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'subsample': subsample,
        'colsample_bytree': colsample,
        'min_child_weight': 3,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }


def train_diverse_ensemble(n_models: int = 7, random_seed: int = RANDOM_SEED):
    print("Loading data and extracting features...")
    battles = load_battle_data(DATA_FILE)
    X_df, y = extract_features_and_labels(battles)

    # Split for scaling and evaluation
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_df, X_test_df)

    model_names: List[str] = []

    """Lightweight shim: implementation moved to `ensemble_three.archived`.

    Top-level module now re-exports the archived implementation so imports
    continue to work while the top-level directory stays tidy.
    """

    from .archived.train_ensemble_diverse import *

    __all__ = [name for name in dir() if not name.startswith('_')]
        model = XGBClassifier(**params)
