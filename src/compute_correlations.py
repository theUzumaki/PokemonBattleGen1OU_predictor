#!/usr/bin/env python3
"""
compute_correlations.py

Compute Pearson and Spearman correlations vs `player_won` for numeric features.
Writes CSV and JSON summary to the analysis output directory.

Usage:
  python3 -m src.compute_correlations --features artifacts/features/features.csv --out artifacts/feature_analysis --top-n 10

"""
import argparse
from pathlib import Path
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", "-f", required=True)
    parser.add_argument("--out", "-o", default="artifacts/feature_analysis")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
        import numpy as np
    except Exception:
        raise SystemExit("Please install pandas and numpy: pip install pandas numpy")

    # optional SciPy for p-values
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        stats = None
        have_scipy = False

    df = pd.read_csv(args.features)
    if "player_won" not in df.columns:
        raise SystemExit("features CSV must contain 'player_won' column")

    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise SystemExit("No numeric columns found in features CSV")

    # ensure player_won numeric
    numeric["player_won"] = numeric["player_won"].astype(float)

    rows = []
    label = numeric["player_won"].values
    for col in [c for c in numeric.columns if c != "player_won"]:
        vals = numeric[col].values
        # mask NaNs
        mask = ~pd.isna(vals)
        x = label[mask]
        y = vals[mask]
        n = int(len(x))
        non_zero_pct = float((y != 0).sum()) / max(1, n) if n > 0 else 0.0

        pear_r = pear_p = None
        spear_r = spear_p = None
        try:
            if n >= 3 and have_scipy:
                pear_r, pear_p = stats.pearsonr(x, y)
                spear_r, spear_p = stats.spearmanr(x, y)
            elif n >= 3:
                # fallback estimates (no p-values)
                pear_r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
                spear_r = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            else:
                pear_r = spear_r = None
        except Exception:
            pear_r = pear_p = spear_r = spear_p = None

        rows.append({
            "feature": col,
            "n": n,
            "pearson_r": float(pear_r) if pear_r is not None else None,
            "pearson_p": float(pear_p) if pear_p is not None else None,
            "spearman_rho": float(spear_r) if spear_r is not None else None,
            "spearman_p": float(spear_p) if spear_p is not None else None,
            "non_zero_pct": non_zero_pct,
        })

    out_df = pd.DataFrame(rows)
    # sort by abs pearson descending
    out_df["abs_pearson"] = out_df["pearson_r"].abs()
    out_df = out_df.sort_values(by="abs_pearson", ascending=False).drop(columns=["abs_pearson"])

    # Write a canonical per-feature table that includes bivariate stats and group means.
    # This mirrors the richer `feature_bivariance_player_won.csv` used elsewhere.
    # Add group means and delta_mean similar to feature_analysis's output.
    try:
        import pandas as _pd
        # compute mean_when_player_won / lost using the original numeric df
        means = []
        for rec in out_df["feature"]:
            col = rec
            vals = numeric[col]
            mw = float(vals[numeric["player_won"] == 1].mean()) if (numeric["player_won"] == 1).any() else None
            ml = float(vals[numeric["player_won"] == 0].mean()) if (numeric["player_won"] == 0).any() else None
            delta = mw - ml if (mw is not None and ml is not None) else None
            means.append((mw, ml, delta))
        out_df["mean_when_player_won"] = [m[0] for m in means]
        out_df["mean_when_player_lost"] = [m[1] for m in means]
        out_df["delta_mean"] = [m[2] for m in means]

        csv_path = out_dir / "feature_bivariance_player_won.csv"
        out_df.to_csv(csv_path, index=False)
    except Exception:
        # fallback: still write the simpler top_correlations.csv if something goes wrong
        csv_path = out_dir / "top_correlations.csv"
        out_df.to_csv(csv_path, index=False)

    # JSON summary: top-n by abs pearson and by abs spearman
    # prepare selection safely when NaNs present
    tmp = out_df.copy()
    tmp["abs_pearson"] = tmp["pearson_r"].abs()
    tmp["abs_spearman"] = tmp["spearman_rho"].abs()
    top_by_pear = tmp.sort_values(by="abs_pearson", ascending=False).head(args.top_n)
    top_by_spear = tmp.sort_values(by="abs_spearman", ascending=False).head(args.top_n)

    summary = {
        "n_features": int(len(out_df)),
        "n_samples": int(numeric.shape[0]),
        "top_by_pearson": top_by_pear.to_dict(orient="records"),
        "top_by_spearman": top_by_spear.to_dict(orient="records"),
        "scipy_available": have_scipy,
    }
    # write summary JSON (keeps the top lists for quick programmatic access)
    with open(out_dir / "top_correlations.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # print a small human-friendly table for top-n by pearson
    print(f"Top {args.top_n} features by |Pearson r| w.r.t player_won:")
    for i, row in enumerate(summary["top_by_pearson"], start=1):
        feat = row["feature"]
        print(f"{i}. {feat}: pearson_r={row['pearson_r']}, pearson_p={row.get('pearson_p')}, spearman_rho={row.get('spearman_rho')}, n={row['n']}")

    print("Wrote:")
    print(f" - {csv_path}")
    print(f" - {out_dir / 'top_correlations.json'}")


if __name__ == '__main__':
    main()
