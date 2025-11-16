"""
Configuration file for Ensemble Three pipeline

Centralized configuration for reproducibility and easy tuning.
"""

import numpy as np
from pathlib import Path

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ============================================================================
# PATHS
# ============================================================================

# Data paths
DATA_DIR = Path('../data')
TRAIN_FILE = DATA_DIR / 'train.jsonl'
TEST_FILE = DATA_DIR / 'test.jsonl'

# Output paths
MODEL_DIR = Path('./models')
RESULTS_DIR = Path('./results')
PREDICTIONS_DIR = Path('./predictions')

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
PREDICTIONS_DIR.mkdir(exist_ok=True)


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Train/test split
TEST_SIZE = 0.2
STRATIFY = True  # Maintain class balance in splits

# Feature preprocessing
SCALER_TYPE = 'standard'  # 'standard' or 'minmax'
HANDLE_MISSING = 'zero'   # How to handle NaN values
HANDLE_INFINITE = 'zero'  # How to handle inf values


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model selection
MODEL_TYPE = 'xgboost'  # Currently supports: 'xgboost'

# XGBoost default parameters
XGBOOST_PARAMS = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'min_child_weight': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': RANDOM_SEED,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
}

# Hyperparameter tuning
TUNE_HYPERPARAMETERS = False  # Set to True for grid search
TUNING_CV_FOLDS = 5
TUNING_SCORING = 'roc_auc'
TUNING_N_JOBS = -1  # Use all CPU cores

# Hyperparameter search space
PARAM_GRID = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Cross-validation
CV_FOLDS = 5
CV_SCORING = 'roc_auc'

# Metrics to compute
COMPUTE_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'roc_auc',
]

# Visualization settings
PLOT_DPI = 300
PLOT_STYLE = 'whitegrid'
TOP_N_FEATURES = 20  # Number of features to show in importance plot


# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================

# Batch processing
BATCH_SIZE = 1000  # For large-scale prediction
SHOW_PROGRESS = True  # Display progress bars

# Output format
PREDICTION_FORMAT = 'csv'  # 'csv' or 'json'


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# Timeline processing
MAX_TURNS = 30  # Only use first N turns

# Type effectiveness
USE_TYPE_EFFECTIVENESS = True

# Derived ratios
COMPUTE_MOMENTUM_INDEX = True
COMPUTE_BATTLE_CONTROL = True

# Feature thresholds
TANK_HP_THRESHOLD = 100
TANK_DEF_THRESHOLD = 80
FAST_SPEED_THRESHOLD = 100
SLOW_SPEED_THRESHOLD = 60


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

VERBOSE = True  # Print detailed progress
LOG_FILE = None  # Set to path for file logging


# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Memory optimization
LOW_MEMORY_MODE = False  # Trade speed for memory

# Parallel processing
N_JOBS = -1  # Number of parallel jobs (-1 = all cores)

# Model persistence
COMPRESS_MODEL = 3  # Compression level for joblib (0-9)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """
    Validate configuration parameters.
    """
    assert 0 < TEST_SIZE < 1, "TEST_SIZE must be between 0 and 1"
    assert CV_FOLDS >= 2, "CV_FOLDS must be at least 2"
    assert MODEL_TYPE in ['xgboost'], f"Unsupported model type: {MODEL_TYPE}"
    assert SCALER_TYPE in ['standard', 'minmax'], f"Unsupported scaler: {SCALER_TYPE}"
    assert TOP_N_FEATURES > 0, "TOP_N_FEATURES must be positive"
    
    if VERBOSE:
        print("✓ Configuration validated successfully")


if __name__ == '__main__':
    print("Ensemble Three Configuration")
    print("=" * 60)
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Model Type: {MODEL_TYPE}")
    print(f"Hyperparameter Tuning: {TUNE_HYPERPARAMETERS}")
    print(f"Cross-Validation Folds: {CV_FOLDS}")
    print(f"Test Size: {TEST_SIZE:.0%}")
    print("=" * 60)
    
    validate_config()
