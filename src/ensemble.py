#!/usr/bin/env python3
"""Create an ensemble prediction from multiple model prediction CSVs.

Reads CSVs with columns `battle_id,player_won` (binary 0/1). Aligns on
`battle_id`, computes mean prediction across provided model files and
produces a final `player_won` prediction using threshold 0.5. In case of
an exact tie (mean == 0.5) the prediction from the XGBoost file
(`predictions_xgboost.csv`) is used as a tiebreaker if present, else the
first file's prediction is used.

Writes `predictions_ensemble.csv` and `predictions_ensemble_probs.csv` by default.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd


def read_pred_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "battle_id" not in df.columns or "player_won" not in df.columns:
        raise ValueError(f"Prediction file {path} must contain 'battle_id' and 'player_won' columns")
    return df.rename(columns={"player_won": path.stem})


def build_ensemble(files, out_csv, out_probs_csv):
    paths = [Path(f) for f in files]
    dfs = [read_pred_csv(p) for p in paths]
    # Merge on battle_id
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="battle_id", how="inner")

    model_cols = [p.stem for p in paths]

    # Ensure predictions are numeric (0/1)
    for c in model_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype(int)

    # Compute mean probability
    merged["player_won_prob"] = merged[model_cols].mean(axis=1)

    # Default ensemble: threshold 0.5. For exact tie (0.5), use xgboost file if present
    def resolve_row(row):
        prob = row["player_won_prob"]
        if prob > 0.5:
            return 1
        if prob < 0.5:
            return 0
        # exact tie -> try xgboost tiebreaker
        xgb_keys = [k for k in model_cols if "xgboost" in k.lower() or "xgb" in k.lower()]
        if xgb_keys:
            return int(row[xgb_keys[0]])
        # fallback to first model
        return int(row[model_cols[0]])

    merged["player_won"] = merged.apply(resolve_row, axis=1).astype(int)

    # Write outputs
    out_csv = Path(out_csv)
    out_probs_csv = Path(out_probs_csv) if out_probs_csv else None

    merged[["battle_id", "player_won"]].to_csv(out_csv, index=False)
    if out_probs_csv:
        merged[["battle_id", "player_won_prob"]].to_csv(out_probs_csv, index=False)

    return out_csv, out_probs_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ensemble binary predictions from multiple CSV files")
    parser.add_argument("files", nargs="*",
                        help="Prediction CSV files to ensemble. Default: predictions_knn.csv predictions_logistic.csv predictions_svm.csv predictions_xgboost.csv",
                        )
    parser.add_argument("--out", default="predictions_ensemble.csv", help="Output CSV path (default: predictions_ensemble.csv)")
    parser.add_argument("--probs-out", default="predictions_ensemble_probs.csv", help="Output probs CSV path (default: predictions_ensemble_probs.csv)")

    args = parser.parse_args(argv)

    if not args.files:
        args.files = [
            "predictions_knn.csv",
            "predictions_logistic.csv",
            "predictions_svm.csv",
            "predictions_xgboost.csv",
        ]

    out_csv, out_probs_csv = build_ensemble(args.files, args.out, args.probs_out)
    print(f"Wrote ensemble predictions to {out_csv}")
    if out_probs_csv:
        print(f"Wrote ensemble probabilities to {out_probs_csv}")


if __name__ == "__main__":
    main()
