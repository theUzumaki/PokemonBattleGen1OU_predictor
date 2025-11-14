#!/usr/bin/env python3
"""
train.py

Small training helper that:
- loads a features CSV (expects a `player_won` column)
- drops sparse features below a non-zero threshold
- exposes `standardize_train_test` to fit a StandardScaler on train and transform train/test
- trains quick baseline models (logistic regression + random forest) and saves models/scaler

Usage:
  python -m src.train --features artifacts/features/features.csv --out artifacts/models

The primary exported functions you can import from this module are:
- drop_sparse_features(df, threshold=0.01, target_col='player_won') -> (df_reduced, dropped)
- standardize_train_test(X_train, X_test=None) -> (X_train_s, X_test_s, scaler)

"""

import argparse
from pathlib import Path
import json
import sys


# NOTE: per user request, sparse-feature dropping logic has been removed.
# If you want to inspect sparsity, use pandas value_counts() or run the
# previous drop-sparse helper offline. Training will use all numeric features
# present in the provided CSV (including near-constant / sparse ones).


def standardize_train_test(X_train, X_test=None):
    """Fit a StandardScaler on X_train and transform X_train and optional X_test.

    Returns (X_train_scaled, X_test_scaled_or_None, scaler).
    X inputs can be pandas DataFrame or numpy arrays; outputs preserve numpy arrays.
    """
    try:
        import numpy as np
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise RuntimeError("scikit-learn and numpy are required for standardization") from e

    # Accept pandas DataFrame or numpy array
    is_df = hasattr(X_train, "columns")
    if is_df:
        cols = X_train.columns.tolist()
        Xtr = X_train.values.astype(float)
    else:
        Xtr = X_train.astype(float)
        cols = None

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)

    Xte_s = None
    if X_test is not None:
        if is_df and hasattr(X_test, "columns"):
            Xte = X_test.values.astype(float)
        else:
            Xte = X_test.astype(float)
        Xte_s = scaler.transform(Xte)

    return Xtr_s, Xte_s, scaler


def train_baselines(X_train, y_train, X_test, y_test, out_dir: Path, feature_names=None):
    """Train a logistic regression and random forest, evaluate and save models.
    Returns metrics dict and saves models and feature importances to out_dir.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        import joblib
    except Exception as e:
        raise RuntimeError("scikit-learn and joblib are required to train models") from e

    out_dir.mkdir(parents=True, exist_ok=True)

    # Make runs more deterministic: fix seeds and limit parallel BLAS threads
    try:
        import os
        import random
        import numpy as np
        # compromise: allow small amount of threading for speed but keep seeds
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
        os.environ.setdefault("PYTHONHASHSEED", "42")
        random.seed(42)
        np.random.seed(42)
    except Exception:
        # If numpy isn't available yet, continue; models will still set random_state
        pass

    results = {}

    # Helper to evaluate and record metrics for a fitted model
    def _eval_model(name, model):
        try:
            ypred_test = model.predict(X_test)
            yprob_test = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
            acc_test = accuracy_score(y_test, ypred_test)
            try:
                auc_test = float(roc_auc_score(y_test, yprob_test)) if yprob_test is not None else None
            except Exception:
                auc_test = None

            ypred_train = model.predict(X_train)
            yprob_train = model.predict_proba(X_train)[:, 1] if hasattr(model, "predict_proba") else None
            acc_train = accuracy_score(y_train, ypred_train)
            try:
                auc_train = float(roc_auc_score(y_train, yprob_train)) if yprob_train is not None else None
            except Exception:
                auc_train = None

            results[name] = {
                "train": {
                    "accuracy": float(acc_train),
                    "roc_auc": auc_train,
                    "report": classification_report(y_train, ypred_train, output_dict=True),
                },
                "test": {
                    "accuracy": float(acc_test),
                    "roc_auc": auc_test,
                    "report": classification_report(y_test, ypred_test, output_dict=True),
                },
            }
        except Exception:
            results[name] = {"error": "evaluation failed"}

    # 1) Logistic Regression
    log = LogisticRegression(max_iter=1000, random_state=42)
    log.fit(X_train, y_train)
    joblib.dump(log, out_dir / "logistic.joblib")
    _eval_model("logistic", log)

    # 2) XGBoost (required)
    xgb_model = None
    try:
        from xgboost import XGBClassifier
        # Use a small number of threads for speed/stability tradeoff
        xgb_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=2, tree_method="hist")
        xgb_model.fit(X_train, y_train)
        joblib.dump(xgb_model, out_dir / "xgboost.joblib")
        _eval_model("xgboost", xgb_model)
    except Exception as e:
        raise RuntimeError("XGBoost is required for training the requested models. Please install xgboost.") from e

    # 3) SVM (probability=True for ROC). Use default RBF kernel.
    svm_model = SVC(probability=True, random_state=42)
    svm_model.fit(X_train, y_train)
    joblib.dump(svm_model, out_dir / "svm.joblib")
    _eval_model("svm", svm_model)

    # 4) K-Nearest Neighbors
    # Let KNN use a couple of jobs to speed up permutation scoring
    knn = KNeighborsClassifier(n_jobs=2)
    knn.fit(X_train, y_train)
    joblib.dump(knn, out_dir / "knn.joblib")
    _eval_model("knn", knn)

    # Permutation importances for each trained model
    from sklearn.inspection import permutation_importance
    perm_results = {}
    models_to_check = {"logistic": log, "xgboost": xgb_model, "svm": svm_model, "knn": knn}
    for mname, m in models_to_check.items():
        try:
            # Compromise: moderate repeats and limited parallelism for better speed
            perm = permutation_importance(m, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2)
            mean = perm.importances_mean
            std = perm.importances_std
            if feature_names is not None:
                perm_map = {feat: {"mean": float(mean[i]), "std": float(std[i])} for i, feat in enumerate(feature_names)}
            else:
                perm_map = {str(i): {"mean": float(v), "std": 0.0} for i, v in enumerate(mean)}
            perm_results[mname] = perm_map
        except Exception:
            perm_results[mname] = {}

    # Save consolidated permutation importances for all models
    try:
        import json
        with open(out_dir / "feature_importances.json", "w", encoding="utf-8") as fh:
            json.dump({"permutation_importance": perm_results}, fh, indent=2)
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(description="Drop sparse features, standardize and run quick baselines")
    parser.add_argument("--features", "-f", required=True, help="Path to features CSV (must contain player_won)")
    parser.add_argument("--out", "-o", default="artifacts/models", help="Output directory to save models and scaler")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        import joblib
    except Exception:
        print("Please install pandas and scikit-learn to run training: pip install pandas scikit-learn joblib")
        raise

    feats = Path(args.features)
    if not feats.exists():
        raise SystemExit(f"Features CSV not found: {feats}")

    df = pd.read_csv(feats)
    if "player_won" not in df.columns:
        raise SystemExit("features CSV must contain 'player_won' column")

    # Keep numeric columns only (player_won included)
    numeric = df.select_dtypes(include=["number"]).copy()

    # Per request: do NOT drop sparse features here. Use all numeric columns.
    reduced = numeric
    X = reduced.drop(columns=["player_won"]) if "player_won" in reduced.columns else reduced
    y = reduced["player_won"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, stratify=y, random_state=args.random_state)

    X_train_s, X_test_s, scaler = standardize_train_test(X_train, X_test)

    # save scaler
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        joblib.dump(scaler, out_dir / "scaler.joblib")
        print(f"Saved scaler to {out_dir / 'scaler.joblib'}")
    except Exception:
        print("Failed to save scaler (joblib missing)")

    # Train quick baselines and save models
    feature_names = X.columns.tolist() if hasattr(X, "columns") else None
    results = train_baselines(X_train_s, y_train.values, X_test_s, y_test.values, out_dir, feature_names=feature_names)

    # write metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print("Training complete. Metrics written to:", out_dir / "metrics.json")

    # --- Also generate a compact summary file (sum_importances.json) so the
    # pipeline can remain self-contained: include per-model train/test accuracy
    # and top/bottom-3 features derived from permutation importances.
    try:
        imps_path = out_dir / "feature_importances.json"
        imps_all = {}
        if imps_path.exists():
            try:
                with open(imps_path, "r", encoding="utf-8") as fh:
                    imps_all = json.load(fh)
            except Exception:
                imps_all = {}

        if isinstance(imps_all, dict) and "permutation_importance" in imps_all:
            imps_map = imps_all.get("permutation_importance", {}) or {}
        else:
            imps_map = imps_all or {}

        model_summaries = {}
        best_model = None
        best_acc = None

        # Use the metrics we just wrote to pick best model and populate accuracies
        for mname, res in (results or {}).items():
            train_acc = None
            test_acc = None
            try:
                if isinstance(res, dict):
                    if "train" in res and isinstance(res["train"], dict):
                        train_acc = res["train"].get("accuracy")
                    if "test" in res and isinstance(res["test"], dict):
                        test_acc = res["test"].get("accuracy")
            except Exception:
                pass

            # update best
            try:
                if test_acc is not None:
                    ta = float(test_acc)
                    if best_acc is None or ta > best_acc:
                        best_acc = ta
                        best_model = mname
            except Exception:
                pass

            # importances
            imps_for_model = imps_map.get(mname, {}) if isinstance(imps_map, dict) else {}
            scored = []
            for feat, v in (imps_for_model or {}).items():
                try:
                    if isinstance(v, dict):
                        mean = float(v.get("mean", 0.0))
                        std = float(v.get("std", 0.0))
                    else:
                        mean = float(v)
                        std = 0.0
                except Exception:
                    mean = 0.0
                    std = 0.0
                scored.append((feat, mean, std))

            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            top3 = [{"feature": f, "mean": m, "std": s} for f, m, s in scored_sorted[:3]]
            bottom3 = [{"feature": f, "mean": m, "std": s} for f, m, s in scored_sorted[-3:]] if scored_sorted else []

            model_summaries[mname] = {
                "train_accuracy": float(train_acc) if train_acc is not None else None,
                "test_accuracy": float(test_acc) if test_acc is not None else None,
                "top_3_features": top3,
                "bottom_3_features": bottom3,
            }

        # include any models present in imps_map but not in results
        for mname, mdict in (imps_map or {}).items():
            if mname in model_summaries:
                continue
            scored = []
            for feat, v in (mdict or {}).items():
                try:
                    if isinstance(v, dict):
                        mean = float(v.get("mean", 0.0))
                        std = float(v.get("std", 0.0))
                    else:
                        mean = float(v)
                        std = 0.0
                except Exception:
                    mean = 0.0
                    std = 0.0
                scored.append((feat, mean, std))
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            top3 = [{"feature": f, "mean": m, "std": s} for f, m, s in scored_sorted[:3]]
            bottom3 = [{"feature": f, "mean": m, "std": s} for f, m, s in scored_sorted[-3:]] if scored_sorted else []
            model_summaries[mname] = {
                "train_accuracy": None,
                "test_accuracy": None,
                "top_3_features": top3,
                "bottom_3_features": bottom3,
            }

        summary_imp = {
            "best_model_by_test_accuracy": best_model,
            "best_model_test_accuracy": best_acc,
            "models": model_summaries,
        }

        try:
            with open(out_dir / "sum_importances.json", "w", encoding="utf-8") as fh:
                json.dump(summary_imp, fh, indent=2)
            print("Wrote sum_importances.json to:", out_dir / "sum_importances.json")
        except Exception:
            print("Failed to write sum_importances.json from train.py")
    except Exception:
        # non-fatal: continue
        pass


if __name__ == '__main__':
    main()
