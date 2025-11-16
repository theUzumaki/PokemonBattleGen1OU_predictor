"""
Compare a single XGBoost model vs the diverse bagging ensemble.

Moved to archived to reduce top-level file noise.
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve
from xgboost import XGBClassifier

from ..train import load_battle_data, extract_features_and_labels, preprocess_features

RESULTS_DIR = Path('./results')
RESULTS_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path('./models')
META_FILE = MODEL_DIR / 'ensemble_meta_diverse.json'

RANDOM_SEED = 42


def train_single_model(X_train_scaled, y_train):
    params = {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'random_state': RANDOM_SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'tree_method': 'hist'
    }
    model = XGBClassifier(**params)
    model.fit(X_train_scaled, y_train, verbose=False)
    return model


def load_diverse_ensemble_models():
    if not META_FILE.exists():
        raise FileNotFoundError(f"Diverse ensemble metadata not found at {META_FILE}")
    with open(META_FILE, 'r') as f:
        meta = json.load(f)
    models = []
    for name in meta.get('models', []):
        p = MODEL_DIR / name
        models.append(joblib.load(p))
    return models


def evaluate_ensemble_on_test(models, X_test_scaled, y_test):
    probas = np.vstack([m.predict_proba(X_test_scaled)[:, 1] for m in models])
    avg_proba = np.mean(probas, axis=0)
    y_pred = (avg_proba > 0.5).astype(int)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, avg_proba))
    }
    return metrics, y_pred, avg_proba


def main():
    battles = load_battle_data(Path('../data/train.jsonl'))
    X_df, y = extract_features_and_labels(battles)
    from sklearn.model_selection import train_test_split
    X_train_df, X_test_df, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train_df, X_test_df)
    single_model = train_single_model(X_train_scaled, y_train)
    single_metrics = {
        'accuracy': float(accuracy_score(y_test, single_model.predict(X_test_scaled))),
        'precision': float(precision_score(y_test, single_model.predict(X_test_scaled))),
        'recall': float(recall_score(y_test, single_model.predict(X_test_scaled))),
        'roc_auc': float(roc_auc_score(y_test, single_model.predict_proba(X_test_scaled)[:, 1]))
    }
    ensemble_models = load_diverse_ensemble_models()
    ensemble_metrics, ensemble_pred, ensemble_proba = evaluate_ensemble_on_test(ensemble_models, X_test_scaled, y_test)
    out = {'single': single_metrics, 'diverse_ensemble': ensemble_metrics}
    with open(RESULTS_DIR / 'ensemble_vs_diverse_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
