#!/usr/bin/env python3
"""
feature_analysis.py (player_won bivariate)

This script focuses only on bivariate analysis between the label `player_won`
and every other numeric feature in the supplied features CSV. It computes
Pearson and Spearman correlations (and p-values if SciPy is available),
writes a CSV `feature_bivariance_player_won.csv` and optionally creates per-
feature plots saved under `plots/player_won/`.

Usage:
  python -m src.feature_analysis --features artifacts/features/features.csv --out artifacts/feature_analysis --plot-top-k 30

Only pandas, numpy, matplotlib, seaborn are required for plotting; SciPy
is optional for p-values.
"""

import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser(description="Bivariate analysis vs player_won")
    parser.add_argument("--features", "-f", required=True, help="Path to features CSV (must include player_won)")
    parser.add_argument("--out", "-o", default="artifacts/feature_analysis", help="Output directory")
    parser.add_argument("--plot-top-k", type=int, default=None, help="If set, only plot top-K features by abs(pearson) (default: plot all)")
    parser.add_argument("--models-dir", type=str, default=None, help="Optional models output dir containing feature_importances.json to identify most important feature")
    parser.add_argument("--max-plots", type=int, default=200, help="Maximum number of plots to write (safety)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots" / "player_won"
    plots_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set(style="whitegrid")
    except Exception:
        print("Please install pandas, matplotlib and seaborn: pip install pandas matplotlib seaborn")
        raise

    features_path = Path(args.features)
    if not features_path.exists():
        raise SystemExit(f"Features file not found: {features_path}")

    df = pd.read_csv(features_path)

    if "player_won" not in df.columns:
        raise SystemExit("Input features CSV must contain a 'player_won' column")

    # keep only numeric-like columns for analysis
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise SystemExit("No numeric features found in the features CSV")

    # ensure player_won is numeric (0/1)
    numeric["player_won"] = numeric["player_won"].astype(float)

    # --- Correlation heatmap (new) -------------------------------------------------
    # compute correlation matrix for all numeric features (including player_won)
    try:
        corr = numeric.corr(method="pearson")
        # save correlation matrix as CSV for downstream inspection
        corr.to_csv(out_dir / "correlation_matrix.csv")

        # plot heatmap (size scales with number of features, but capped)
        n_cols = len(corr.columns)
        # sensible figure size: 0.35 per feature up to a max to avoid enormous figures
        fig_size = (min(28, max(6, 0.35 * n_cols)), min(28, max(6, 0.35 * n_cols)))
        try:
            fig, ax = plt.subplots(figsize=fig_size)
            # mask the upper triangle for readability
            import numpy as _np
            mask = _np.triu(_np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap="vlag", center=0, annot=False, fmt=".2f",
                        square=False, linewidths=0.25, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Feature correlation matrix (pearson)")
            fig.tight_layout()
            fig.savefig(out_dir / "correlation_heatmap.png", dpi=150)
            plt.close(fig)
        except Exception:
            # if plotting fails, continue without aborting the script
            try:
                plt.close('all')
            except Exception:
                pass
    except Exception:
        # correlation is a non-essential plot; proceed silently if it fails
        pass
    # -----------------------------------------------------------------------------

    # prepare SciPy if available
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        stats = None
        have_scipy = False

    results = []
    label = numeric["player_won"].values
    cols = [c for c in numeric.columns if c != "player_won"]

    for col in cols:
        y = numeric[col].values
        mask = ~pd.isna(y)
        y2 = y[mask]
        x2 = label[mask]
        n = int(len(y2))
        if n < 3:
            pear_r = None
            pear_p = None
            spear_r = None
            spear_p = None
        else:
            try:
                if have_scipy:
                    pear_r, pear_p = stats.pearsonr(x2, y2)
                    spear_r, spear_p = stats.spearmanr(x2, y2)
                else:
                    # fallbacks
                    pear_r = float(np.corrcoef(x2, y2)[0, 1]) if np.std(x2) > 0 and np.std(y2) > 0 else 0.0
                    pear_p = None
                    spear_r = float(pd.Series(x2).corr(pd.Series(y2), method="spearman"))
                    spear_p = None
            except Exception:
                pear_r = None
                pear_p = None
                spear_r = None
                spear_p = None

        # group stats
        try:
            mean_when_won = float(np.nanmean(y2[x2 == 1])) if (x2 == 1).any() else None
            mean_when_lost = float(np.nanmean(y2[x2 == 0])) if (x2 == 0).any() else None
        except Exception:
            mean_when_won = None
            mean_when_lost = None

        delta_mean = None
        if mean_when_won is not None and mean_when_lost is not None:
            delta_mean = mean_when_won - mean_when_lost

        non_zero_pct = float(((y2 != 0).sum()) / max(1, len(y2))) if n > 0 else 0.0

        results.append({
            "feature": col,
            "n": n,
            "pearson_r": float(pear_r) if pear_r is not None else None,
            "pearson_p": float(pear_p) if pear_p is not None else None,
            "spearman_rho": float(spear_r) if spear_r is not None else None,
            "spearman_p": float(spear_p) if spear_p is not None else None,
            "mean_when_player_won": mean_when_won,
            "mean_when_player_lost": mean_when_lost,
            "delta_mean": delta_mean,
            "non_zero_pct": non_zero_pct,
        })

    res_df = pd.DataFrame(results)
    # sort by absolute pearson correlation descending (NaNs go last)
    res_df["abs_pearson"] = res_df["pearson_r"].abs()
    res_df = res_df.sort_values(by="abs_pearson", ascending=False).drop(columns=["abs_pearson"])
    res_df.to_csv(out_dir / "feature_bivariance_player_won.csv", index=False)

    # If models dir provided, try to identify the single most important feature
    most_imp_feature = None
    if args.models_dir is not None:
        try:
            import json as _json
            from pathlib import Path as _P
            imp_path = _P(args.models_dir) / "feature_importances.json"
            if imp_path.exists():
                with open(imp_path, "r", encoding="utf-8") as fh:
                    imps = _json.load(fh)
                # Expect imps to contain 'logistic' and 'random_forest' dicts mapping feature->value
                log_map = imps.get("logistic", {}) or {}
                rf_map = imps.get("random_forest", {}) or {}
                # Build combined score: normalize abs(logistic coef) and rf importance to [0,1] then sum
                feats = sorted(set(list(log_map.keys()) + list(rf_map.keys())))
                if feats:
                    import numpy as _np
                    log_vals = _np.array([abs(float(log_map.get(f, 0.0))) for f in feats], dtype=float)
                    rf_vals = _np.array([float(rf_map.get(f, 0.0)) for f in feats], dtype=float)
                    log_norm = log_vals / (log_vals.max() if log_vals.max() > 0 else 1.0)
                    rf_norm = rf_vals / (rf_vals.max() if rf_vals.max() > 0 else 1.0)
                    combined = log_norm + rf_norm
                    top_idx = int(_np.argmax(combined))
                    most_imp_feature = feats[top_idx]
        except Exception:
            most_imp_feature = None

    # plotting
    plot_limit = args.plot_top_k or len(res_df)
    plot_limit = min(plot_limit, args.max_plots)
    plotted = 0

    for idx, row in res_df.head(plot_limit).iterrows():
        col = row["feature"]
        y = numeric[col]
        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            # For numeric vs binary label, a boxplot by label is informative
            sns.boxplot(x=numeric["player_won"].astype(int), y=y, ax=ax)
            sns.stripplot(x=numeric["player_won"].astype(int), y=y, color="0.2", size=3, jitter=0.15, ax=ax)
            ax.set_xlabel("player_won")
            ax.set_ylabel(col)
            ax.set_title(f"{col} vs player_won (pearson={row['pearson_r']:.3f}" if row["pearson_r"] is not None else f"{col} vs player_won")
        except Exception:
            # fallback: simple scatter with jitter on x
            xs = numeric["player_won"].astype(float) + (np.random.rand(len(numeric)) - 0.5) * 0.08
            ax.scatter(xs, y, s=8, alpha=0.6)
            ax.set_xlabel("player_won")
            ax.set_ylabel(col)

        safe_name = col.replace("/", "_").replace(" ", "_")
        try:
            fig.tight_layout()
            fig.savefig(plots_dir / f"{safe_name}.png")
            plt.close(fig)
            plotted += 1
        except Exception:
            plt.close(fig)
            continue

    summary = {
        "n_features_analyzed": len(cols),
        "n_samples": int(len(numeric)),
        "plots_written": plotted,
        "scipy_available": have_scipy,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("Bivariate analysis (player_won) complete. Outputs written to:", out_dir)

    # If a models dir was provided, write a compact summary of importances
    # selecting the model with the best test accuracy and logging its top/bottom 3 features
    if args.models_dir is not None:
        try:
            mdir = Path(args.models_dir)
            metrics_path = mdir / "metrics.json"
            imps_path = mdir / "feature_importances.json"

            # Load metrics
            metrics = {}
            if metrics_path.exists():
                try:
                    with open(metrics_path, "r", encoding="utf-8") as fh:
                        metrics = json.load(fh)
                except Exception:
                    metrics = {}

            # Load importances (support both wrapped and flat formats)
            imps_all = {}
            if imps_path.exists():
                try:
                    with open(imps_path, "r", encoding="utf-8") as fh:
                        imps_all = json.load(fh)
                except Exception:
                    imps_all = {}

            # extract permutation map if wrapped
            if isinstance(imps_all, dict) and "permutation_importance" in imps_all:
                imps_map = imps_all.get("permutation_importance", {}) or {}
            else:
                imps_map = imps_all or {}

            # Build per-model summaries
            model_summaries = {}
            best_model = None
            best_acc = None

            # Determine best model by test accuracy from metrics.json
            for mname, minfo in (metrics or {}).items():
                train_acc = None
                test_acc = None
                try:
                    if isinstance(minfo, dict):
                        if "train" in minfo and isinstance(minfo["train"], dict):
                            train_acc = minfo["train"].get("accuracy")
                        if "test" in minfo and isinstance(minfo["test"], dict):
                            test_acc = minfo["test"].get("accuracy")
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

                # gather importances for this model if present
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

            # If there are models present in imps_map but not in metrics, include them too
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

            # If an ensemble prediction file exists at repo root, compute basic ensemble metrics
            try:
                from pathlib import Path as _P
                import sklearn.metrics as _m
                repo_root = Path.cwd()
                ens_csv = _P(repo_root) / "predictions_ensemble.csv"
                ens_probs = _P(repo_root) / "predictions_ensemble_probs.csv"
                ensemble_info = {}
                if ens_csv.exists():
                    try:
                        ens_df = pd.read_csv(ens_csv)
                        # Ensure required columns
                        if "battle_id" in ens_df.columns and "player_won" in ens_df.columns:
                            # align to labels in `numeric` (features file rows)
                            y_true = numeric["player_won"].astype(int).reset_index(drop=True)
                            # build prediction vector aligned by battle_id
                            ypred = pd.Series(index=y_true.index, dtype="float64")
                            for _, r in ens_df.iterrows():
                                try:
                                    bid = int(r["battle_id"])
                                    if 0 <= bid < len(y_true):
                                        ypred.iloc[bid] = int(r["player_won"])
                                except Exception:
                                    continue
                            mask = ~ypred.isna()
                            if mask.any():
                                y_true_masked = y_true[mask].values
                                ypred_masked = ypred[mask].astype(int).values
                                acc = float(_m.accuracy_score(y_true_masked, ypred_masked))
                                roc = None
                                # if probability file exists, try to compute ROC AUC
                                if ens_probs.exists():
                                    try:
                                        pdf = pd.read_csv(ens_probs)
                                        prob_ser = pd.Series(index=y_true.index, dtype="float64")
                                        # find prob column
                                        prob_col = None
                                        for c in pdf.columns:
                                            if c == "player_won_prob" or c.endswith("_prob") or c == "prob":
                                                prob_col = c
                                                break
                                        if prob_col is None:
                                            cols = [c for c in pdf.columns if c != "battle_id"]
                                            if len(cols) == 1:
                                                prob_col = cols[0]
                                        if prob_col is not None:
                                            for _, rr in pdf.iterrows():
                                                try:
                                                    bid = int(rr["battle_id"])
                                                    if 0 <= bid < len(y_true):
                                                        prob_ser.iloc[bid] = float(rr[prob_col])
                                                except Exception:
                                                    continue
                                            prob_mask = ~prob_ser.isna() & mask
                                            if prob_mask.any():
                                                yprob = prob_ser[prob_mask].astype(float).values
                                                ytrue_for_prob = y_true[prob_mask].values
                                                roc = float(_m.roc_auc_score(ytrue_for_prob, yprob))
                                    except Exception:
                                        roc = None
                                ensemble_info = {"accuracy": acc, "roc_auc": roc}
                    except Exception:
                        ensemble_info = {}
                if ensemble_info:
                    summary_imp["ensemble"] = ensemble_info
            except Exception:
                # non-fatal: continue without ensemble entry
                pass

            # write file
            try:
                with open(mdir / "sum_importances.json", "w", encoding="utf-8") as fh:
                    json.dump(summary_imp, fh, indent=2)
                print("Wrote importance summary to:", mdir / "sum_importances.json")
            except Exception:
                print("Failed to write sum_importances.json")
        except Exception as exc:
            print("Error while generating sum_importances.json:", exc)
    # If we identified a most important feature, produce an extended analysis
    if most_imp_feature is not None:
        try:
            feat = most_imp_feature
            y = numeric[feat]
            # compute basic stats
            pear = float(res_df.loc[res_df.feature == feat, "pearson_r"].values[0]) if (res_df.feature == feat).any() else None
            spear = float(res_df.loc[res_df.feature == feat, "spearman_rho"].values[0]) if (res_df.feature == feat).any() else None
            delta = float(res_df.loc[res_df.feature == feat, "delta_mean"].values[0]) if (res_df.feature == feat).any() else None

            # load importances for reporting
            imp_path = Path(args.models_dir) / "feature_importances.json"
            imp_info = {}
            if imp_path.exists():
                with open(imp_path, "r", encoding="utf-8") as fh:
                    imp_info = json.load(fh)

            logistic_val = None
            rf_val = None
            if isinstance(imp_info, dict):
                logistic_val = imp_info.get("logistic", {}).get(feat)
                rf_val = imp_info.get("random_forest", {}).get(feat)

            # plot violin + strip
            try:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.violinplot(x=numeric["player_won"].astype(int), y=y, inner=None, ax=ax)
                sns.boxplot(x=numeric["player_won"].astype(int), y=y, width=0.12, showcaps=True, boxprops={'facecolor':'white'}, ax=ax)
                sns.stripplot(x=numeric["player_won"].astype(int), y=y, color="0.2", size=3, jitter=0.15, ax=ax)
                ax.set_xlabel("player_won")
                ax.set_ylabel(feat)
                ax.set_title(f"Most important: {feat}")
                fig.tight_layout()
                fig.savefig(plots_dir / f"most_important__{feat}.png")
                plt.close(fig)
            except Exception:
                pass

            # write summary json
            extended = {
                "most_important_feature": feat,
                "pearson_r": pear,
                "spearman_rho": spear,
                "delta_mean": delta,
                "logistic_value": logistic_val,
                "random_forest_value": rf_val,
            }
            with open(out_dir / f"most_important_feature_summary.json", "w", encoding="utf-8") as fh:
                json.dump(extended, fh, indent=2)
        except Exception:
            pass


if __name__ == '__main__':
    main()
