"""
Testing utilities and example tests (archived).
"""

import json
from pathlib import Path

from ..feature_extractor import extract_all_features
from ..predict import predict_win_probability


def create_example_battle() -> dict:
    # Minimal example preserved
    return {
        'player_won': True,
        'battle_id': 999,
        'p1_team_details': [],
        'p2_lead_details': {},
        'battle_timeline': []
    }


def test_feature_extraction():
    battle = create_example_battle()
    features = extract_all_features(battle)
    return True


def test_prediction():
    model_path = Path('./models/xgboost_model.joblib')
    if not model_path.exists():
        return False
    battle = create_example_battle()
    try:
        _ = predict_win_probability(battle)
        return True
    except Exception:
        return False


def run_all_tests():
    results = []
    results.append(('Feature Extraction', test_feature_extraction()))
    results.append(('Prediction', test_prediction()))
    for name, passed in results:
        print(f"{name}: {'PASS' if passed else 'FAIL'}")


if __name__ == '__main__':
    run_all_tests()
