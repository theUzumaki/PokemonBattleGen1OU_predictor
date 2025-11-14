Feature correlation helper

This repository includes `src/feature_correlation.py` — a script to extract numeric features
from a JSONL battles dataset and compute correlations among them.

Quick start

1. (Optional) Create a venv and install dependencies:

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install pandas seaborn matplotlib

2. Run the script on a subset for a fast check:

   python -m src.feature_correlation --input data/train.jsonl --output out --max-records 100

Outputs placed in `out/`:
- `features_table.csv` — extracted features per record
- `correlation_pearson.csv`, `correlation_spearman.csv` — correlation matrices
- `correlation_<method>_heatmap.png` — saved heatmaps (unless disabled)
- `top_corr_<method>.txt` — top correlated feature pairs

Notes

- The script extracts team-level aggregate stats (mean/sum/min/max of base stats),
  lead stats, timeline-derived features (num_turns, avg move base_power), and any
  top-level numeric scalars it finds.
- Adjust `--max-records` for quick iteration when the dataset is large.
