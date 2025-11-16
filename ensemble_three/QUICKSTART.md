# Ensemble Three - Quick Start Guide

Get up and running with the Pokémon battle prediction pipeline in 5 minutes.

## Installation

```bash
# Navigate to the ensemble_three directory
cd ensemble_three

# Install dependencies
pip install -r requirements.txt
```

## Step 1: Test Feature Extraction

Verify that feature extraction works correctly:

```bash
python feature_extractor.py
```

**Expected output:** A list of ~70 extracted features with their values.

## Step 2: Train the Model

Train the XGBoost model on your data:

```bash
python train.py
```

**What happens:**
- Loads training data from `../data/train.jsonl`
- Extracts features from all battles
- Trains XGBoost classifier with 80/20 train/test split
- Evaluates with accuracy, precision, recall, ROC-AUC
- Performs 5-fold cross-validation
- Saves model to `./models/`
- Generates plots in `./results/`

**Duration:** ~2-5 minutes for 1000 battles (depends on dataset size)

**Output files:**
```
models/
  ├── xgboost_model.joblib       # Trained model
  ├── feature_scaler.joblib       # Feature scaler
  ├── feature_names.joblib        # Feature names
  └── metadata.json               # Training metadata

results/
  ├── confusion_matrix.png        # Confusion matrix heatmap
  ├── roc_curve.png               # ROC curve plot
  ├── feature_importance.png      # Top features bar chart
  └── feature_importance.csv      # All feature importances
```

## Step 3: Make Predictions

### Option A: Predict on Test Data

```bash
python predict.py
```

This automatically processes `../data/test.jsonl` and saves predictions to `./predictions/test_predictions.csv`.

### Option B: Predict Programmatically

```python
from predict import predict_win_probability

# Your battle data
battle = {
    'p1_team_details': [...],
    'p2_lead_details': {...},
    'battle_timeline': [...]
}

# Get prediction
probability = predict_win_probability(battle)
print(f"Win probability: {probability:.2%}")
```

## Step 4: Run Tests

Verify everything works:

```bash
python test_pipeline.py
```

**Tests run:**
1. Feature extraction
2. Single battle prediction
3. Batch prediction

## Quick Troubleshooting

### "Model not found" error
- **Solution:** Run `python train.py` first

### "No module named 'xgboost'" error
- **Solution:** `pip install -r requirements.txt`

### "File not found" for train.jsonl
- **Solution:** Ensure data files are in `../data/` directory

### Training is too slow
- **Solution:** Set `tune_hyperparameters=False` in `train.py` (line 140)

### Out of memory during training
- **Solution:** Reduce dataset size or use a machine with more RAM

## Configuration

Edit `config.py` to customize:
- Random seed
- Hyperparameter search space
- Train/test split ratio
- Model parameters

## Next Steps

1. **Review Results:** Check plots in `./results/` to understand model performance
2. **Feature Importance:** Look at `feature_importance.csv` to see what matters most
3. **Tune Model:** Enable hyperparameter tuning for better accuracy (slower)
4. **Deploy:** Use `predict.py` module in your application

## Example Usage

```python
from predict import predict_win_probability, predict_batch

# Single prediction
battle = load_battle_from_somewhere()
prob = predict_win_probability(battle)

if prob > 0.6:
    print("Strong win predicted!")
elif prob > 0.5:
    print("Slight win predicted")
elif prob > 0.4:
    print("Close battle, slight loss")
else:
    print("Strong loss predicted")

# Batch predictions
battles = [battle1, battle2, battle3, ...]
probabilities = predict_batch(battles)
```

## Performance Benchmarks

On a typical laptop:
- **Feature extraction:** ~0.001s per battle
- **Single prediction:** ~0.002s (after model loaded)
- **Batch prediction:** ~0.001s per battle
- **Training (1000 battles):** ~2 minutes without tuning, ~20 minutes with tuning

## Common Use Cases

### 1. Predict on New Test Set

```bash
# Place your test.jsonl in ../data/
python predict.py
# Check ./predictions/test_predictions.csv
```

### 2. Retrain with Different Parameters

Edit `config.py`:
```python
XGBOOST_PARAMS = {
    'max_depth': 7,        # Deeper trees
    'learning_rate': 0.1,  # Faster learning
    'n_estimators': 300,   # More trees
}
```

Then run:
```bash
python train.py
```

### 3. Feature Engineering

Add custom features in `feature_extractor.py`:

```python
def extract_custom_features(battle):
    features = {}
    # Your custom feature logic here
    features['my_custom_feature'] = some_calculation()
    return features
```

Update `extract_all_features()` to include your function.

### 4. Generate Kaggle Submission

```python
from predict import predict_from_file

# Generate predictions
results = predict_from_file(
    '../data/test.jsonl',
    './predictions/submission.csv'
)

# Format for Kaggle (if needed)
results['battle_id'] = results['battle_id'].astype(int)
results['prediction'] = results['win_probability']
results[['battle_id', 'prediction']].to_csv(
    './predictions/kaggle_submission.csv',
    index=False
)
```

## Getting Help

- **Feature extraction issues:** Check `feature_extractor.py` comments
- **Training issues:** Check `train.py` comments
- **Prediction issues:** Check `predict.py` comments
- **General questions:** Read the main `README.md`

## Summary

**3 commands to get started:**
```bash
pip install -r requirements.txt  # Install dependencies
python train.py                  # Train model
python predict.py                # Make predictions
```

That's it! You now have a working ML pipeline for Pokémon battle prediction.
