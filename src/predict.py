import argparse
import json
# When running as a module (python -m src.predict) use a relative import;
# fall back to the top-level import to support running the file directly.
try:
    from . import feature_extractor
except Exception:
    import feature_extractor
from pathlib import Path
import numpy as np
import joblib
import csv

"""
Loads a model and makes predictions on input data.
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Load a model and optionally predict on input")
    parser.add_argument("-m", "--model", type=str, default="artifacts/models/model.pkl", help="Path to saved model")
    parser.add_argument("-i", "--input", type=str, default="data/test.jsonl", help="JSON array or path to .npy file for prediction")
    args = parser.parse_args()

    default_dir = Path("artifacts/models")

    # Resolve model path robustly: allow user to pass either a full path
    # or a model name (with/without .joblib). Also accept the default
    # value used in the CLI which may already look like a path.
    model_arg = args.model
    model_path = Path(model_arg)
    cand_paths = [model_path]
    if not model_path.exists():
        cand_paths = [
            Path(str(model_arg)),
            Path(str(model_arg) + ".joblib"),
            default_dir / model_arg,
            default_dir / (model_arg + ".joblib"),
        ]
        found = None
        for p in cand_paths:
            if p.exists():
                found = p
                break
        if found is None:
            tried = ", ".join(str(p) for p in cand_paths)
            raise SystemExit(f"Model file not found: tried {tried}")
        model_path = found

    model = joblib.load(model_path)
    print(f"Loaded model from {model_path} (type: {type(model)})")

    # Try to load a scaler from same directory as the model, falling back to `artifacts/models/scaler.joblib`
    scaler = None
    try:
        scaler_path = model_path.parent / "scaler.joblib"
        fallback = default_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        elif fallback.exists():
            scaler = joblib.load(fallback)
            print(f"Loaded scaler from {fallback}")
        else:
            print("No scaler found; predictions will be made without scaling")
    except Exception as e:
        print("Failed to load scaler:", e)

    if args.input:
        inp = Path(args.input)
        if inp.exists():
            with inp.open("r", encoding="utf-8") as f:
                recs = [json.loads(line) for line in f if line.strip()]

            features = []
            for record in recs:
                extracted = feature_extractor.extract_20_features(record)

                # If the extractor returns a dict (it should), remove the label
                # and preserve the insertion order of features so columns align
                # with the CSV produced during training (which used the same
                # extractor to write headers).
                if isinstance(extracted, dict):
                    extracted.pop("player_won", None)
                    # preserve insertion order
                    vals = list(extracted.values())
                else:
                    vals = list(extracted)

                features.append(vals)

            X_features = np.asarray(features, dtype=float)

            # Replace any NaNs or None (defensive) with 0.0 to avoid scaler errors
            # (training used numeric-only CSV; missing values are unexpected)
            X_features = np.nan_to_num(X_features.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

            # Apply scaler if we loaded one
            if scaler is not None:
                try:
                    X_features = scaler.transform(X_features)
                except Exception as e:
                    print("Warning: scaler.transform failed, proceeding without scaling:", e)

            preds = getattr(model, "predict", lambda x: None)(X_features)
            output_path = "predictions_" + args.model + ".csv"
            with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["battle_id", "player_won"])
                for count, pred in enumerate(preds):
                    writer.writerow([count, pred])
            print(f"Predictions saved to {output_path}")