"""Lightweight shim: examples moved to `ensemble_three.archived`.

Keep the top-level package tidy while preserving the usage examples
under `ensemble_three.archived.examples`.
"""

from .archived.examples import *

__all__ = [name for name in dir() if not name.startswith('_')]
    
    # Get predictions
    predictions = predict_batch(battles)
    actuals = [1 if b['player_won'] else 0 for b in battles]
    
    # Find errors
    errors = []
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        predicted_class = 1 if pred > 0.5 else 0
        if predicted_class != actual:
            errors.append({
                'battle_id': battles[i].get('battle_id', i),
                'predicted': pred,
                'actual': actual,
                'error_type': 'False Positive' if predicted_class == 1 else 'False Negative'
            })
    
    error_df = pd.DataFrame(errors)
    
    print(f"\nAnalyzed {len(battles)} battles")
    print(f"Errors: {len(errors)} ({len(errors)/len(battles)*100:.1f}%)")
    
    if len(errors) > 0:
        print("\nError Types:")
        print(error_df['error_type'].value_counts())
        
        print("\nWorst Errors (by confidence):")
        error_df['confidence'] = abs(error_df['predicted'] - 0.5)
        print(error_df.sort_values('confidence', ascending=False).head())


# =============================================================================
# MAIN MENU
# =============================================================================

def main():
    """
    Interactive example menu.
    """
    print("\n" + "="*80)
    print("ENSEMBLE THREE - USAGE EXAMPLES")
    print("="*80)
    print("\nAvailable examples:")
    print("  1. Training a Model")
    print("  2. Single Battle Prediction")
    print("  3. Batch Prediction from File")
    print("  4. Feature Extraction and Analysis")
    print("  5. Adding Custom Features")
    print("  6. Model Interpretation")
    print("  7. Hyperparameter Tuning")
    print("  8. Kaggle Submission")
    print("  9. Real-time Battle Prediction")
    print("  10. Error Analysis")
    print("  0. Run all examples")
    
    choice = input("\nSelect example (0-10): ")
    
    examples = {
        '1': example_training,
        '2': example_single_prediction,
        '3': example_batch_prediction,
        '4': example_feature_analysis,
        '5': example_custom_features,
        '6': example_model_interpretation,
        '7': example_hyperparameter_tuning,
        '8': example_kaggle_submission,
        '9': example_realtime_prediction,
        '10': example_error_analysis,
    }
    
    if choice == '0':
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"\n⚠ Error in example: {e}")
    elif choice in examples:
        examples[choice]()
    else:
        print("Invalid choice")


if __name__ == '__main__':
    main()
