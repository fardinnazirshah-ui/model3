"""
src/utils.py
Utility functions for metrics calculation, plotting, and validation.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, f1_score,
    roc_auc_score, matthews_corrcoef, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def calculate_metrics(y_actual, y_pred, model_name='Model'):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_actual: Actual labels
        y_pred: Predicted labels
        model_name: Name of model for logging
    
    Returns:
        Dict of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_actual, y_pred).ravel()
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_actual, y_pred),
        'recall': recall_score(y_actual, y_pred, zero_division=0),
        'precision': precision_score(y_actual, y_pred, zero_division=0),
        'f1': f1_score(y_actual, y_pred, zero_division=0),
        'matthews_cc': matthews_corrcoef(y_actual, y_pred),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }
    
    logger.info(f"{model_name} Metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    
    return metrics


def plot_confusion_matrix(
    y_actual, y_pred, model_name='Model',
    output_path=None
):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_actual: Actual labels
        y_pred: Predicted labels
        model_name: Model name for title
        output_path: Where to save plot (optional)
    """
    cm = confusion_matrix(y_actual, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Normal', 'Recession'],
        yticklabels=['Normal', 'Recession']
    )
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {output_path}")
    
    plt.close()


def validate_data_shapes(
    X: np.ndarray,
    y: np.ndarray,
    expected_features: int = None
) -> bool:
    """
    Validate data shapes and consistency.
    
    Args:
        X: Feature matrix
        y: Target vector
        expected_features: Expected number of features (optional)
    
    Returns:
        True if valid, False otherwise
    """
    # Check dimensions match
    if len(X) != len(y):
        logger.error(f"Shape mismatch: X has {len(X)} samples, y has {len(y)}")
        return False
    
    # Check feature count
    if expected_features and X.shape[1] != expected_features:
        logger.error(
            f"Feature count mismatch: expected {expected_features}, "
            f"got {X.shape[1]}"
        )
        return False
    
    # Check for NaN
    if np.isnan(X).any() or np.isnan(y).any():
        logger.error("Data contains NaN values")
        return False
    
    # Check class distribution
    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    
    logger.info(f"✓ Data validation passed: X.shape={X.shape}, y.shape={y.shape}")
    return True


def create_feature_importance_df(
    importances: np.ndarray,
    feature_names: list,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Create and sort feature importance dataframe.
    
    Args:
        importances: Array of feature importances
        feature_names: List of feature names
        top_n: Number of top features to return
    
    Returns:
        DataFrame sorted by importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Top {top_n} features by importance:")
    for idx, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    return importance_df


if __name__ == "__main__":
    # Test utilities
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=100, n_features=20, n_informative=10)
    y_pred = np.random.randint(0, 2, 100)
    
    print("Testing metrics calculation...")
    metrics = calculate_metrics(y, y_pred, "Test Model")
    print(metrics)
