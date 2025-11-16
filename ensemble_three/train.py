"""
Training Pipeline for Pokémon Battle Prediction

Complete training pipeline including:
- Data loading and preprocessing
- Feature extraction and scaling
- Model training with XGBoost
- Hyperparameter tuning via GridSearchCV
- Model evaluation and visualization
- Model persistence
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from .feature_extractor import extract_all_features, get_feature_names


# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = Path('../data')
TRAIN_FILE = DATA_DIR / 'train.jsonl'
MODEL_DIR = Path('./models')
RESULTS_DIR = Path('./results')

# Ensure directories exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def load_battle_data(file_path: Path) -> List[Dict]:
    """
    Load battle data from JSONL file.
    
    Args:
        file_path: Path to .jsonl file
        
    Returns:
        List of battle dictionaries
    """
    battles = []
    print(f"Loading data from {file_path}...")
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                battle = json.loads(line.strip())
                battles.append(battle)
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(battles)} battles")
    return battles


def extract_features_and_labels(battles: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract features and labels from battle data.
    
    Args:
        battles: List of battle dictionaries
        
    Returns:
        Tuple of (features DataFrame, labels array)
    """
    print("Extracting features from battles...")
    
    feature_list = []
    labels = []
    
    for i, battle in enumerate(battles):
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(battles)} battles...")
        
        try:
            features = extract_all_features(battle)
            feature_list.append(features)
            labels.append(1 if battle['player_won'] else 0)
        except Exception as e:
            print(f"Warning: Could not extract features from battle {battle.get('battle_id', i)}: {e}")
            continue
    
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(feature_list)
    labels = np.array(labels)
    
    print(f"Extracted {len(df)} feature vectors with {len(df.columns)} features each")
    print(f"Label distribution: {np.sum(labels)} wins, {len(labels) - np.sum(labels)} losses")
    print(f"Win rate: {np.mean(labels):.2%}")
    
    return df, labels


def preprocess_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess features: handle missing values and scale.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (scaled X_train, scaled X_test, fitted scaler)
    """
    print("Preprocessing features...")
    
    # Handle missing values (fill with 0)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    # Handle infinite values
    X_train = X_train.replace([np.inf, -np.inf], 0)
    X_test = X_test.replace([np.inf, -np.inf], 0)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Feature preprocessing complete")
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Test shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, scaler


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune_hyperparameters: bool = True
) -> XGBClassifier:
    """
    Train XGBoost classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        tune_hyperparameters: Whether to perform grid search
        
    Returns:
        Trained XGBClassifier model
    """
    print("\nTraining XGBoost model...")
    
    if tune_hyperparameters:
        print("Performing hyperparameter tuning with GridSearchCV...")
        
        # Define parameter grid
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }
        
        # Base model
        base_model = XGBClassifier(
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation ROC-AUC: {grid_search.best_score_:.4f}")
        
        model = grid_search.best_estimator_
        
    else:
        # Use default good parameters
        print("Training with default parameters...")
        model = XGBClassifier(
            max_depth=5,
            learning_rate=0.05,
            n_estimators=200,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
            use_label_encoder=False,
            eval_metric='logloss',
            tree_method='hist'
        )
        
        model.fit(X_train, y_train)
    
    print("Model training complete")
    return model


def evaluate_model(
    model: XGBClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Evaluate model performance and generate visualizations.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model performance...")
    
    # Training set predictions
    y_train_pred = model.predict(X_train)
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_pred_proba)
    
    # Test set predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    metrics = {
        'train_accuracy': train_accuracy,
        'train_precision': train_precision,
        'train_recall': train_recall,
        'train_roc_auc': train_roc_auc,
        'test_accuracy': test_accuracy,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_roc_auc': test_roc_auc,
    }
    
    # Print results
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print("\nTRAINING SET:")
    print(f"  Accuracy:  {train_accuracy:.4f}")
    print(f"  Precision: {train_precision:.4f}")
    print(f"  Recall:    {train_recall:.4f}")
    print(f"  ROC-AUC:   {train_roc_auc:.4f}")
    print("\nVALIDATION SET (Test):")
    print(f"  Accuracy:  {test_accuracy:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall:    {test_recall:.4f}")
    print(f"  ROC-AUC:   {test_roc_auc:.4f}")
    print("="*80)
    
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualizations
    create_evaluation_plots(
        y_test, y_pred, y_pred_proba, cm, model, feature_names
    )
    
    return metrics


def create_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    cm: np.ndarray,
    model: XGBClassifier,
    feature_names: List[str]
):
    """
    Create and save evaluation visualizations.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        cm: Confusion matrix
        model: Trained model
        feature_names: List of feature names
    """
    print("\nGenerating evaluation plots...")
    
    # Set style
    sns.set_style('whitegrid')
    
    # 1. Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Loss', 'Win'],
                yticklabels=['Loss', 'Win'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved confusion matrix plot")
    
    # 2. ROC Curve
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=16, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved ROC curve plot")
    
    # 3. Feature Importance (Top 20)
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    top_n = 20
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved feature importance plot")
    
    # Save feature importance to CSV
    importance_df.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)
    print("  ✓ Saved feature importance CSV")


def save_model_artifacts(
    model: XGBClassifier,
    scaler: StandardScaler,
    feature_names: List[str],
    metrics: Dict[str, float]
):
    """
    Save trained model and associated artifacts.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        feature_names: List of feature names
        metrics: Evaluation metrics
    """
    print("\nSaving model artifacts...")
    
    # Save model
    model_path = MODEL_DIR / 'xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"  ✓ Saved model to {model_path}")
    
    # Save scaler
    scaler_path = MODEL_DIR / 'feature_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler to {scaler_path}")
    
    # Save feature names
    feature_names_path = MODEL_DIR / 'feature_names.joblib'
    joblib.dump(feature_names, feature_names_path)
    print(f"  ✓ Saved feature names to {feature_names_path}")
    
    # Save metadata
    metadata = {
        'random_seed': RANDOM_SEED,
        'num_features': len(feature_names),
        'metrics': metrics,
        'model_type': 'XGBClassifier',
    }
    metadata_path = MODEL_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata to {metadata_path}")


def main():
    """
    Main training pipeline.
    """
    print("="*80)
    print("POKÉMON BATTLE PREDICTION - TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load data
    battles = load_battle_data(TRAIN_FILE)
    
    # 2. Extract features and labels
    X, y = extract_features_and_labels(battles)
    
    # Get feature names
    feature_names = list(X.columns)
    
    # 3. Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  Train size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Test size:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    
    # 4. Preprocess features
    X_train_scaled, X_test_scaled, scaler = preprocess_features(X_train, X_test)
    
    # 5. Train model
    # Set tune_hyperparameters=True for grid search (takes longer)
    # Set to False for faster training with good default parameters
    model = train_xgboost_model(
        X_train_scaled, 
        y_train, 
        tune_hyperparameters=False  # Change to True for hyperparameter tuning
    )
    
    # 6. Evaluate model
    metrics = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, feature_names)
    
    # 7. Cross-validation score
    print("\nPerforming 5-fold cross-validation on full dataset...")
    cv_scores = cross_val_score(
        model, 
        np.vstack([X_train_scaled, X_test_scaled]),
        np.concatenate([y_train, y_test]),
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    print(f"Cross-validation ROC-AUC scores: {cv_scores}")
    print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 8. Save artifacts
    save_model_artifacts(model, scaler, feature_names, metrics)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nModel artifacts saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nNext steps:")
    print("  1. Review the evaluation plots in the results directory")
    print("  2. Use predict.py to make predictions on new battles")
    print("="*80)


if __name__ == '__main__':
    main()
