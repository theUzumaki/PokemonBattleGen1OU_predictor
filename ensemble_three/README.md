# Ensemble Three: Pokémon Battle Prediction Pipeline

A complete, production-ready machine learning pipeline for predicting Pokémon battle outcomes using gradient boosting with comprehensive feature engineering.

## 🎯 Overview

This pipeline predicts the probability that a player wins a Pokémon battle based on:
- Player's full team composition (6 Pokémon)
- Opponent's lead Pokémon
- Battle dynamics from the first 30 turns

**Key Features:**
- ✅ 100+ engineered features covering team stats, type matchups, and battle momentum
- ✅ XGBoost classifier with hyperparameter tuning support
- ✅ Comprehensive evaluation with visualizations
- ✅ Clean inference API for predictions
- ✅ Fully reproducible with fixed random seeds
- ✅ Well-documented and modular code

## 🧹 Pruned / Archived Files

To improve readability the repository was pruned: several large or
auxiliary scripts were moved to `ensemble_three/archived/` and the
top-level Python modules were replaced with small re-export stubs.
This preserves the original implementations and keeps existing
imports working while making the package directory easier to scan.

If you need any archived file, open `ensemble_three/archived/` where
the full implementations remain available.

## 📊 Feature Engineering

The pipeline extracts 4 categories of predictive features:

### A. Team Composition Features (24 features)
- Aggregate statistics: mean, std, min, max of base HP/Atk/Def/SpA/SpD/Spe
- Type coverage: unique types, type diversity score
- Role distribution: physical/special/balanced attackers, tanks
- Speed tiers: fast/slow Pokémon counts
- Overall team quality metrics

### B. Opponent Lead Features (13 features)
- Opponent lead base stats (HP, Atk, Def, SpA, SpD, Spe)
- Type matchup analysis vs player team
- Advantage/disadvantage counts
- Physical vs special attacker identification

### C. Battle Timeline Features (27 features)
Extracted from first 30 turns:
- HP change differentials (damage dealt vs received)
- KO counts per side
- Move statistics: average power, category distribution
- Status effect success rates
- Speed advantage (who moves first)
- Pokémon diversity (revealed opponents)
- Boost tracking
- Priority move usage

### D. Derived Ratios (6 features)
- Offensive efficiency = damage dealt / damage taken
- Survivability ratio = HP% / turns survived
- Momentum index = (KOs for - against) + (status inflicted - suffered)
- Team vs lead stat ratio
- Speed advantage score
- Battle control composite score

**Total: 70+ features**

## 🚀 Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib
```

### Training

```bash
cd ensemble_three
python train.py
```

This will:
1. Load training data from `../data/train.jsonl`
2. Extract features from all battles
3. Train an XGBoost classifier
4. Evaluate on test set with cross-validation
5. Generate evaluation plots (confusion matrix, ROC curve, feature importance)
6. Save model artifacts to `./models/`

**Training Options:**
- Set `tune_hyperparameters=True` in `train.py` for grid search (slower but better)
- Set `tune_hyperparameters=False` for faster training with good defaults

### Prediction

```python
from predict import predict_win_probability

# Load battle data
battle = {
    'p1_team_details': [...],  # Player's 6 Pokémon
    'p2_lead_details': {...},   # Opponent's lead
    'battle_timeline': [...]    # Up to 30 turns
}

# Get win probability
probability = predict_win_probability(battle)
print(f"Win probability: {probability:.2%}")
```

**Batch Prediction:**

```bash
python predict.py
```

This will automatically process `../data/test.jsonl` and save predictions to `./predictions/test_predictions.csv`.

## 📁 Project Structure

```
ensemble_three/
├── __init__.py              # Package initialization
├── feature_extractor.py     # Feature engineering module
├── train.py                 # Training pipeline
├── predict.py               # Inference module
├── config.py                # Configuration parameters
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── models/                  # Trained model artifacts
│   ├── xgboost_model.joblib
│   ├── feature_scaler.joblib
│   ├── feature_names.joblib
│   └── metadata.json
├── results/                 # Evaluation results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── feature_importance.csv
└── predictions/             # Prediction outputs
    └── test_predictions.csv
```

## 🔬 Feature Extraction Details

### Type Effectiveness System

The pipeline implements a comprehensive type effectiveness chart for calculating offensive/defensive matchups. This enables features like:
- Number of team members with type advantage vs opponent lead
- Net type advantage score
- Offensive coverage metrics

### Battle Timeline Processing

For each of the first 30 turns, we track:
- **HP Changes**: Calculate damage dealt/received for each Pokémon
- **KO Events**: Count fainted Pokémon per side
- **Move Analysis**: Power, category (Physical/Special/Status), priority
- **Status Effects**: Track status inflictions and their success
- **Stat Boosts**: Sum of all stat stage changes
- **Turn Order**: Determine speed advantage

### Missing Data Handling

- Battles with < 30 turns: Features computed from available turns
- Missing timeline data: Zero-filled with appropriate defaults
- Infinite values: Replaced with 0 during preprocessing
- NaN values: Filled with 0

## 📈 Model Architecture

**Algorithm:** XGBoost (Gradient Boosting)

**Why XGBoost?**
- Excellent performance on structured/tabular data
- Built-in feature importance
- Robust to outliers and missing values
- Fast training and inference
- Interpretable results

**Default Hyperparameters:**
```python
max_depth=5              # Tree depth
learning_rate=0.05       # Shrinkage
n_estimators=200         # Number of trees
min_child_weight=3       # Minimum sum of instance weight
subsample=0.9            # Row sampling ratio
colsample_bytree=0.9     # Column sampling ratio
```

**Tunable Hyperparameters (Grid Search):**
- `max_depth`: [3, 5, 7]
- `learning_rate`: [0.01, 0.05, 0.1]
- `n_estimators`: [100, 200, 300]
- `min_child_weight`: [1, 3, 5]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]

## 📊 Evaluation Metrics

The pipeline reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate among predicted wins
- **Recall**: True positive rate among actual wins
- **ROC-AUC**: Area under the ROC curve (primary metric)
- **Confusion Matrix**: Visual breakdown of predictions
- **5-Fold Cross-Validation**: Robust performance estimate

## 🎨 Visualizations

Generated plots include:

1. **Confusion Matrix**: Shows true positives, false positives, true negatives, false negatives
2. **ROC Curve**: Visualizes model's discrimination ability across thresholds
3. **Feature Importance**: Bar chart of top 20 most important features

All plots saved to `./results/` at 300 DPI for publication quality.

## 🔧 Configuration

Edit `config.py` to customize:
- Random seed for reproducibility
- Train/test split ratio
- Hyperparameter search space
- Feature scaling method
- Model save paths

## 📝 API Reference

### `predict_win_probability(battle_json: dict) -> float`

**Main inference function.**

**Args:**
- `battle_json`: Dictionary with keys `p1_team_details`, `p2_lead_details`, `battle_timeline`

**Returns:**
- Float between 0 and 1 representing win probability

**Example:**
```python
prob = predict_win_probability(battle)
if prob > 0.5:
    print(f"Player favored to win ({prob:.1%} confidence)")
else:
    print(f"Opponent favored to win ({1-prob:.1%} confidence)")
```

### `predict_batch(battle_list: list) -> np.ndarray`

**Batch prediction for multiple battles.**

**Args:**
- `battle_list`: List of battle dictionaries

**Returns:**
- NumPy array of win probabilities

**Example:**
```python
battles = [battle1, battle2, battle3]
probs = predict_batch(battles)
```

### `predict_from_file(input_file: str, output_file: str = None)`

**Load battles from JSONL and generate predictions.**

**Args:**
- `input_file`: Path to .jsonl file
- `output_file`: Optional CSV output path

**Returns:**
- DataFrame with battle_id and win_probability

## 🧪 Testing

Test feature extraction:
```bash
python feature_extractor.py
```

Test prediction pipeline:
```bash
python predict.py
```

## 🚀 Performance Tips

1. **Faster Training**: Set `tune_hyperparameters=False` in `train.py`
2. **Better Accuracy**: Enable hyperparameter tuning (takes ~10x longer)
3. **Large Datasets**: Use batch prediction instead of single predictions
4. **Memory**: Feature extraction is vectorized and memory-efficient

## 📚 Data Format

### Input Battle JSON

```json
{
  "player_won": true,
  "p1_team_details": [
    {
      "name": "starmie",
      "level": 100,
      "types": ["psychic", "water"],
      "base_hp": 60,
      "base_atk": 75,
      "base_def": 85,
      "base_spa": 100,
      "base_spd": 100,
      "base_spe": 115
    }
    // ... 5 more Pokémon
  ],
  "p2_lead_details": {
    // Same structure as above
  },
  "battle_timeline": [
    {
      "turn": 1,
      "p1_pokemon_state": {
        "name": "starmie",
        "hp_pct": 1.0,
        "status": "nostatus",
        "effects": ["noeffect"],
        "boosts": {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0}
      },
      "p1_move_details": {
        "name": "icebeam",
        "type": "ICE",
        "category": "SPECIAL",
        "base_power": 95,
        "accuracy": 1.0,
        "priority": 0
      },
      "p2_pokemon_state": { /* ... */ },
      "p2_move_details": { /* ... */ }
    }
    // ... up to 30 turns
  ],
  "battle_id": 10
}
```

## 🤝 Contributing

This is a complete, self-contained ML pipeline. To extend:

1. **Add Features**: Edit `feature_extractor.py` and add to `extract_*_features()` functions
2. **Try New Models**: Replace XGBClassifier in `train.py` with RandomForest, LightGBM, etc.
3. **Tune Further**: Expand hyperparameter grid in `train_xgboost_model()`
4. **Add Ensembles**: Combine multiple models using `VotingClassifier` or stacking

## 📄 License

MIT License - feel free to use for academic or commercial purposes.

## 👨‍💻 Author

Built by a data science expert specializing in competitive gaming analytics.

## 🙏 Acknowledgments

- XGBoost team for the excellent gradient boosting library
- scikit-learn for the ML infrastructure
- Pokémon battle mechanics community for type effectiveness data

---

**For questions or issues, please check the code comments or raise an issue.**
