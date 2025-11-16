"""
Prediction Module for Pokémon Battle Outcome

Provides clean inference function to predict win probability from battle JSON data.
Loads trained model artifacts and applies feature extraction pipeline.
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Union

from .feature_extractor import extract_all_features


# Paths
MODEL_DIR = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'xgboost_model.joblib'
SCALER_PATH = MODEL_DIR / 'feature_scaler.joblib'
FEATURE_NAMES_PATH = MODEL_DIR / 'feature_names.joblib'

# Global variables for lazy loading
_model = None  # either single model or list of models when ensemble present
_scaler = None
_feature_names = None

# Ensemble cache
_is_ensemble = False



def load_model_artifacts():
    """
    Load trained model, scaler, and feature names.
    
    Uses lazy loading pattern - only loads once on first call.
    """
    global _model, _scaler, _feature_names
    
    global _model, _scaler, _feature_names, _is_ensemble

    if _model is None:
        print("Loading model artifacts...")

        # First, check for ensemble metadata
        ensemble_meta = MODEL_DIR / 'ensemble_meta.json'
        if ensemble_meta.exists():
            print(f"  ✓ Found ensemble metadata at {ensemble_meta}")
            with open(ensemble_meta, 'r') as f:
                meta = json.load(f)

            models_list = []
            for mname in meta.get('models', []):
                mpath = MODEL_DIR / mname
                if not mpath.exists():
                    raise FileNotFoundError(f"Ensemble model file missing: {mpath}")
                models_list.append(joblib.load(mpath))

            _model = models_list
            _is_ensemble = True
            # Load scaler and feature names
            _scaler = joblib.load(SCALER_PATH)
            _feature_names = joblib.load(FEATURE_NAMES_PATH)

            print(f"  ✓ Loaded ensemble of {len(models_list)} models")
            print(f"  ✓ Loaded scaler from {SCALER_PATH}")
            print(f"  ✓ Loaded feature names ({len(_feature_names)} features)")

        else:
            # Fallback to single model
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found at {MODEL_PATH}. Please run train.py or train_ensemble.py first."
                )

            _model = joblib.load(MODEL_PATH)
            _scaler = joblib.load(SCALER_PATH)
            _feature_names = joblib.load(FEATURE_NAMES_PATH)

            print(f"  ✓ Loaded model from {MODEL_PATH}")
            print(f"  ✓ Loaded scaler from {SCALER_PATH}")
            print(f"  ✓ Loaded feature names ({len(_feature_names)} features)")

    return _model, _scaler, _feature_names


def predict_win_probability(battle_json: Dict[str, Any]) -> float:
    """
    Predict the probability that the player wins the battle.
    
    This is the main inference function. Given a battle JSON record,
    it extracts features, scales them, and returns the predicted
    probability of player victory.
    
    Args:
        battle_json: Dictionary containing battle data with keys:
            - p1_team_details: List of player's 6 Pokémon
            - p2_lead_details: Opponent's lead Pokémon
            - battle_timeline: List of turn dictionaries (up to 30)
            
    Returns:
        Float between 0 and 1 representing win probability
        
    Example:
        >>> battle = {
        ...     'p1_team_details': [...],
        ...     'p2_lead_details': {...},
        ...     'battle_timeline': [...]
        ... }
        >>> prob = predict_win_probability(battle)
        >>> print(f"Win probability: {prob:.2%}")
        Win probability: 73.45%
    """
    # Load model artifacts (lazy loading)
    model, scaler, feature_names = load_model_artifacts()
    
    # Extract features from battle
    features = extract_all_features(battle_json)
    
    # Convert to DataFrame to ensure correct feature ordering
    feature_df = pd.DataFrame([features])
    
    # Ensure all expected features are present
    for fname in feature_names:
        if fname not in feature_df.columns:
            feature_df[fname] = 0
    
    # Reorder columns to match training
    feature_df = feature_df[feature_names]
    
    # Handle missing/infinite values
    feature_df = feature_df.fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # Scale features
    features_scaled = scaler.transform(feature_df)
    
    # Predict probability (support ensemble averaging if model is a list)
    if isinstance(model, list):
        probs = np.vstack([m.predict_proba(features_scaled)[:, 1] for m in model])
        prob = float(np.mean(probs, axis=0)[0])
    else:
        prob = float(model.predict_proba(features_scaled)[0, 1])
    
    return float(prob)


def predict_batch(battle_list: list) -> np.ndarray:
    """
    Predict win probabilities for a batch of battles.
    
    More efficient than calling predict_win_probability in a loop.
    
    Args:
        battle_list: List of battle JSON dictionaries
        
    Returns:
        NumPy array of win probabilities
    """
    # Load model artifacts
    model, scaler, feature_names = load_model_artifacts()
    
    print(f"Extracting features from {len(battle_list)} battles...")
    
    # Extract features for all battles
    feature_list = []
    for i, battle in enumerate(battle_list):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(battle_list)}...")
        
        features = extract_all_features(battle)
        feature_list.append(features)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame(feature_list)
    
    # Ensure all expected features are present
    for fname in feature_names:
        if fname not in feature_df.columns:
            feature_df[fname] = 0
    
    # Reorder columns
    feature_df = feature_df[feature_names]
    
    # Handle missing/infinite values
    feature_df = feature_df.fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # Scale features
    features_scaled = scaler.transform(feature_df)

    # Predict probabilities (average if ensemble)
    if isinstance(model, list):
        probs = np.mean(np.vstack([m.predict_proba(features_scaled)[:, 1] for m in model]), axis=0)
    else:
        probs = model.predict_proba(features_scaled)[:, 1]
    
    return probs


def predict_from_file(input_file: Union[str, Path], output_file: Union[str, Path] = None):
    """
    Load battles from a JSONL file and generate predictions.
    
    Predictions are discrete (0 or 1) based on probability > 0.5 threshold.
    
    Args:
        input_file: Path to input .jsonl file
        output_file: Optional path to save predictions CSV with columns:
                     - battle_id: Battle identifier
                     - player_won: Discrete prediction (0=loss, 1=win)
    
    Note:
        For probability values instead of discrete predictions, use predict_batch()
        and access the raw probabilities directly.
    """
    input_file = Path(input_file)
    
    print(f"Loading battles from {input_file}...")
    battles = []
    battle_ids = []
    
    with open(input_file, 'r') as f:
        for line in f:
            battle = json.loads(line.strip())
            battles.append(battle)
            battle_ids.append(battle.get('battle_id', len(battles) - 1))
    
    print(f"Loaded {len(battles)} battles")
    
    # Predict
    probabilities = predict_batch(battles)
    
    # Create results DataFrame with discrete predictions
    results = pd.DataFrame({
        'battle_id': battle_ids,
        'player_won': (probabilities > 0.5).astype(int)  # Discrete 0 or 1
    })
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"\n✓ Saved predictions to {output_file}")
    else:
        print("\nPredictions:")
        print(results)
    
    return results


def main():
    """
    Example usage of the prediction module.
    """
    import sys
    
    print("="*80)
    print("POKÉMON BATTLE PREDICTION - INFERENCE")
    print("="*80)
    
    # Check if test file exists
    test_file = Path('../data/test.jsonl')
    
    if test_file.exists():
        print(f"\nFound test file: {test_file}")
        print("Generating predictions...\n")
        
        # Create predictions directory
        pred_dir = Path('./predictions')
        pred_dir.mkdir(exist_ok=True)
        
        # Generate predictions
        results = predict_from_file(
            test_file,
            pred_dir / 'test_predictions.csv'
        )
        
        print("\nPrediction Summary:")
        print(f"  Total battles: {len(results)}")
        print(f"  Predicted wins: {results['player_won'].sum()} ({results['player_won'].mean()*100:.1f}%)")
        print(f"  Predicted losses: {(1-results['player_won']).sum()} ({(1-results['player_won']).mean()*100:.1f}%)")
        
    else:
        print(f"\nTest file not found at {test_file}")
        print("\nExample usage:")
        print("  from predict import predict_win_probability")
        print("  ")
        print("  battle = {")
        print("      'p1_team_details': [...],")
        print("      'p2_lead_details': {...},")
        print("      'battle_timeline': [...]")
        print("  }")
        print("  ")
        print("  prob = predict_win_probability(battle)")
        print("  print(f'Win probability: {prob:.2%}')")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
