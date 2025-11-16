"""
Feature Analysis: Correlation and Importance Analysis

This file was moved into `ensemble_three.archived` to reduce top-level clutter.
Original implementation preserved for advanced analysis.
"""

# --- original implementation (preserved) ---
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier

from ..feature_extractor import extract_all_features


def load_sample_data(data_file: Path, max_samples: int = 1000):
    battles = []
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            line = line.strip()
            if line:
                battles.append(json.loads(line))

    feature_list = []
    labels = []
    for battle in battles:
        try:
            features = extract_all_features(battle)
            feature_list.append(features)
            labels.append(1 if battle['player_won'] else 0)
        except Exception:
            continue

    df = pd.DataFrame(feature_list)
    labels = np.array(labels)
    return df, labels


def calculate_target_correlation(X: pd.DataFrame, y: np.ndarray, top_n: int = 20):
    correlations = []
    for col in X.columns:
        if X[col].std() == 0:
            continue
        pearson_corr, pearson_p = pearsonr(X[col], y)
        spearman_corr, spearman_p = spearmanr(X[col], y)
        correlations.append({
            'feature': col,
            'pearson_corr': pearson_corr,
            'pearson_pval': pearson_p,
            'spearman_corr': spearman_corr,
            'spearman_pval': spearman_p,
            'abs_pearson': abs(pearson_corr),
            'abs_spearman': abs(spearman_corr)
        })

    corr_df = pd.DataFrame(correlations).dropna().sort_values('abs_pearson', ascending=False)
    return corr_df


def find_multicollinearity(X: pd.DataFrame, threshold: float = 0.8):
    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    return pd.DataFrame(high_corr_pairs)


def analyze_model_importance():
    model_path = Path('./models/xgboost_model.joblib')
    feature_names_path = Path('./models/feature_names.joblib')
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'importance_pct': importance / importance.sum() * 100
    }).sort_values('importance', ascending=False)
    return importance_df


def plot_correlation_heatmap(X: pd.DataFrame, top_n: int = 30):
    variances = X.var().sort_values(ascending=False)
    top_features = variances.head(top_n).index
    corr_matrix = X[top_features].corr()
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1)
    output_path = Path('./results/correlation_heatmap.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    # Keep a safe, small entrypoint for manual runs
    data_file = Path('../data/train.jsonl')
    if data_file.exists():
        X, y = load_sample_data(data_file, max_samples=500)
        corr_df = calculate_target_correlation(X, y)
        _ = find_multicollinearity(X)
        _ = analyze_model_importance()