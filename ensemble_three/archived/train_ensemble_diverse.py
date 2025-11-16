"""
Train a diverse XGBoost ensemble (archived copy).
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import List

from ..train import load_battle_data, extract_features_and_labels, preprocess_features
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

MODEL_DIR = Path('./models')
MODEL_DIR.mkdir(exist_ok=True)
DATA_FILE = Path('../data/train.jsonl')

RANDOM_SEED = 42


def sample_params(rng: np.random.RandomState):
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
    battles = load_battle_data(DATA_FILE)
    X_df, y = extract_features_and_labels(battles)
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=random_seed, stratify=y)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_df, X_test_df)
    model_names: List[str] = []
    rng = np.random.RandomState(random_seed)
    for i in range(n_models):
        seed = random_seed + i * 7
        params = sample_params(rng)
        params['random_state'] = int(seed)
        model = XGBClassifier(**params)
        indices = rng.choice(X_train_scaled.shape[0], size=X_train_scaled.shape[0], replace=True)
        X_boot = X_train_scaled[indices]
        y_boot = y_train[indices]
        model.fit(X_boot, y_boot, verbose=False)
        model_name = f'xgboost_ensemble_diverse_{i+1}.joblib'
        joblib.dump(model, MODEL_DIR / model_name)
        model_names.append(model_name)
    joblib.dump(scaler, MODEL_DIR / 'feature_scaler.joblib')
    joblib.dump(list(X_df.columns), MODEL_DIR / 'feature_names.joblib')
    meta = {'type': 'xgboost_diverse_bagging_ensemble', 'n_models': n_models, 'models': model_names, 'seed': random_seed}
    with open(MODEL_DIR / 'ensemble_meta_diverse.json', 'w') as f:
        json.dump(meta, f, indent=2)


if __name__ == '__main__':
    train_diverse_ensemble(n_models=7)
