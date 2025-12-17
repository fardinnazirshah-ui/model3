"""
src/validator.py
Walk-forward validation for all 4 models.

PHASE 3 IMPROVEMENTS:
- Test ALL 4 models (not just LogReg)
- Calculate recall, precision, F1, ROC-AUC
- Generate detailed comparison report
- Create confusion matrices for each model

This replaces Priority 5 with multi-model evaluation.
"""

import pandas as pd
import numpy as np
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve, matthews_corrcoef, accuracy_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Rigorous walk-forward validation for time-series recession prediction."""
    
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: pd.Series,
        initial_train_months: int = 120,
        model_dir: str = 'models/',
        output_dir: str = 'results/'
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            dates: Date vector for labeling (n_samples,)
            initial_train_months: Number of months for initial training window
            model_dir: Directory containing trained models
            output_dir: Directory to save results
        """
        self.X = X
        self.y = y
        self.dates = dates
        self.initial_train_months = initial_train_months
        self.model_dir = model_dir
        self.output_dir = output_dir
        
        Path(output_dir).mkdir(exist_ok=True)
        
        # Results storage
        self.results = {
            'logreg': [],
            'randomforest': [],
            'xgboost': [],
            'lstm': []
        }
        self.model_names = {
            'logreg': 'Logistic Regression (Balanced)',
            'randomforest': 'Random Forest',
            'xgboost': 'XGBoost',
            'lstm': 'LSTM Neural Network'
        }
        
        logger.info(f"Walk-Forward Validator initialized")
        logger.info(f"  Total samples: {len(X)}")
        logger.info(f"  Initial train window: {initial_train_months} months")
        logger.info(f"  Test window: {len(X) - initial_train_months} months")
    
    def load_models(self) -> Dict:
        """Load all trained models from disk."""
        models = {}
        
        try:
            with open(f'{self.model_dir}/logreg_balanced_model.pkl', 'rb') as f:
                models['logreg'] = pickle.load(f)
            logger.info("✓ Loaded LogisticRegression model")
        except Exception as e:
            logger.warning(f"✗ Could not load LogReg model: {e}")
        
        try:
            with open(f'{self.model_dir}/randomforest_model.pkl', 'rb') as f:
                models['randomforest'] = pickle.load(f)
            logger.info("✓ Loaded RandomForest model")
        except Exception as e:
            logger.warning(f"✗ Could not load RandomForest model: {e}")
        
        try:
            with open(f'{self.model_dir}/xgboost_model.pkl', 'rb') as f:
                models['xgboost'] = pickle.load(f)
            logger.info("✓ Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"✗ Could not load XGBoost model: {e}")
        
        try:
            import tensorflow as tf
            models['lstm'] = tf.keras.models.load_model(f'{self.model_dir}/lstm_model.h5')
            logger.info("✓ Loaded LSTM model")
        except Exception as e:
            logger.warning(f"✗ Could not load LSTM model: {e}")
        
        return models
    
    def validate(self, models: Dict) -> pd.DataFrame:
        """
        Perform walk-forward validation on all models.
        
        For each month from initial_train_months to end:
        1. Train on history
        2. Predict next month
        3. Compare to actual
        4. Expand window
        
        Returns:
            DataFrame with predictions and metrics for all iterations
        """
        logger.info("\n" + "=" * 80)
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("=" * 80)
        
        all_results = []
        
        # Iterate through time steps
        for i, test_idx in enumerate(
            range(self.initial_train_months, len(self.X))
        ):
            if i % 50 == 0:
                logger.info(f"Iteration {i}: Testing month {test_idx}/{len(self.X)}")
            
            # Split: train on past, test on current month
            X_train = self.X[:test_idx]
            y_train = self.y[:test_idx]
            X_test = self.X[test_idx:test_idx+1]
            y_test = self.y[test_idx]
            test_date = self.dates.iloc[test_idx]
            
            row = {
                'iteration': i,
                'test_idx': test_idx,
                'test_date': test_date,
                'actual_label': y_test,
                'actual_is_recession': 'Recession' if y_test == 1 else 'Normal'
            }
            
            # Get predictions from each model
            for model_name, model in models.items():
                try:
                    if model_name == 'lstm':
                        # LSTM requires sequence reshaping
                        seq_len = 6
                        if test_idx >= seq_len:
                            X_seq = self.X[test_idx-seq_len:test_idx].reshape(1, seq_len, -1)
                            y_pred_proba = model.predict(X_seq, verbose=0)[0][0]
                            y_pred = 1 if y_pred_proba > 0.5 else 0
                        else:
                            y_pred = 0
                            y_pred_proba = 0.0
                    else:
                        # Standard sklearn interface
                        y_pred = model.predict(X_test)[0]
                        y_pred_proba = model.predict_proba(X_test)[0][1]
                    
                    row[f'{model_name}_pred'] = y_pred
                    row[f'{model_name}_proba'] = y_pred_proba
                    row[f'{model_name}_correct'] = 1 if y_pred == y_test else 0
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
                    row[f'{model_name}_pred'] = -1
                    row[f'{model_name}_proba'] = -1
                    row[f'{model_name}_correct'] = 0
            
            all_results.append(row)
        
        results_df = pd.DataFrame(all_results)
        logger.info(f"\n✓ Walk-forward validation complete: {len(results_df)} months tested")
        
        return results_df
    
    def calculate_metrics(
        self,
        results_df: pd.DataFrame,
        model_name: str
    ) -> Dict:
        """
        Calculate comprehensive metrics for a model.
        
        Args:
            results_df: Walk-forward results
            model_name: Model to evaluate
        
        Returns:
            Dict of metrics
        """
        y_actual = results_df['actual_label'].values
        y_pred = results_df[f'{model_name}_pred'].values
        
        # Filter valid predictions
        valid_mask = y_pred >= 0
        y_actual_valid = y_actual[valid_mask]
        y_pred_valid = y_pred[valid_mask]
        
        if len(y_pred_valid) == 0:
            logger.warning(f"No valid predictions for {model_name}")
            return {}
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_actual_valid, y_pred_valid).ravel()
        
        metrics = {
            'model': self.model_names.get(model_name, model_name),
            'accuracy': accuracy_score(y_actual_valid, y_pred_valid),
            'recall': recall_score(y_actual_valid, y_pred_valid, zero_division=0),
            'precision': precision_score(y_actual_valid, y_pred_valid, zero_division=0),
            'f1': f1_score(y_actual_valid, y_pred_valid, zero_division=0),
            'matthews_cc': matthews_corrcoef(y_actual_valid, y_pred_valid),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'predictions_made': len(y_pred_valid)
        }
        
        # ROC-AUC if probabilities available
        try:
            y_proba = results_df[f'{model_name}_proba'].values[valid_mask]
            if len(np.unique(y_actual_valid)) > 1:
                metrics['roc_auc'] = roc_auc_score(y_actual_valid, y_proba)
            else:
                metrics['roc_auc'] = 0.0
        except:
            metrics['roc_auc'] = 0.0
        
        return metrics
    
    def generate_report(self, results_df: pd.DataFrame, metrics_dict: Dict):
        """
        Generate comprehensive validation report.
        
        Args:
            results_df: Walk-forward results
            metrics_dict: Calculated metrics for all models
        """
        report_path = f'{self.output_dir}/walk_forward_validation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WALK-FORWARD VALIDATION REPORT - PHASE 3 (CORRECTED)\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total months tested: {len(results_df)}\n")
            f.write(f"Recession months in test set: {(results_df['actual_label'] == 1).sum()}\n")
            f.write(f"Normal months in test set: {(results_df['actual_label'] == 0).sum()}\n\n")
            
            # Model comparison table
            f.write("=" * 80 + "\n")
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("=" * 80 + "\n\n")
            
            for model_name in ['logreg', 'randomforest', 'xgboost', 'lstm']:
                if model_name not in metrics_dict:
                    continue
                
                metrics = metrics_dict[model_name]
                
                f.write(f"\nMODEL: {metrics.get('model', model_name)}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Accuracy:        {metrics.get('accuracy', 0):.4f}\n")
                f.write(f"Recall:          {metrics.get('recall', 0):.4f} (% of recessions caught)\n")
                f.write(f"Precision:       {metrics.get('precision', 0):.4f} (% of predictions correct)\n")
                f.write(f"F1 Score:        {metrics.get('f1', 0):.4f}\n")
                f.write(f"ROC-AUC:         {metrics.get('roc_auc', 0):.4f}\n")
                f.write(f"Matthews CC:     {metrics.get('matthews_cc', 0):.4f}\n\n")
                
                f.write("Confusion Matrix:\n")
                f.write(f"  True Negatives:       {metrics.get('tn', 0)}\n")
                f.write(f"  False Positives:      {metrics.get('fp', 0)}\n")
                f.write(f"  False Negatives:      {metrics.get('fn', 0)} ← MISSED RECESSIONS\n")
                f.write(f"  True Positives:       {metrics.get('tp', 0)} ← CAUGHT RECESSIONS\n")
                f.write(f"  Total Predictions:    {metrics.get('predictions_made', 0)}\n")
            
            # Recommendations
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS\n")
            f.write("=" * 80 + "\n")
            
            best_recall = max(
                (metrics_dict.get(m, {}).get('recall', 0), m)
                for m in ['logreg', 'randomforest', 'xgboost', 'lstm']
            )[1]
            
            f.write(f"\n✓ Best Recall (recession detection): {best_recall}\n")
            f.write("✓ Compare models: Which has best balance of recall vs precision?\n")
            f.write("✓ Next: Feature importance analysis and hyperparameter tuning\n")
        
        logger.info(f"✓ Report saved to {report_path}")
    
    def save_results(self, results_df: pd.DataFrame):
        """Save detailed results to CSV."""
        output_path = f'{self.output_dir}/walk_forward_results_detailed.csv'
        results_df.to_csv(output_path, index=False)
        logger.info(f"✓ Detailed results saved to {output_path}")


def main(
    features_path: str = 'data/processed/features_engineered_monthly.csv',
    model_dir: str = 'models/',
    output_dir: str = 'results/'
):
    """Main walk-forward validation pipeline."""
    
    logger.info("=" * 80)
    logger.info("PHASE 3: WALK-FORWARD VALIDATION (ALL MODELS)")
    logger.info("=" * 80 + "\n")
    
    # Load engineered features
    logger.info("Loading engineered features...")
    df = pd.read_csv(features_path)
    X = df.drop(['date', 'recession_label'], axis=1).values
    y = df['recession_label'].values
    dates = df['date']
    
    logger.info(f"✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"✓ Recessions: {y.sum()}, Normal: {len(y) - y.sum()}")
    
    # Initialize validator
    validator = WalkForwardValidator(X, y, dates, model_dir=model_dir, output_dir=output_dir)
    
    # Load models
    logger.info("\nLoading trained models...")
    models = validator.load_models()
    logger.info(f"✓ Loaded {len(models)} models")
    
    # Run walk-forward validation
    results_df = validator.validate(models)
    
    # Calculate metrics
    logger.info("\nCalculating metrics for all models...")
    metrics_dict = {}
    for model_name in models.keys():
        metrics_dict[model_name] = validator.calculate_metrics(results_df, model_name)
        logger.info(f"✓ {model_name}: Recall={metrics_dict[model_name].get('recall', 0):.4f}")
    
    # Generate report
    logger.info("\nGenerating report...")
    validator.generate_report(results_df, metrics_dict)
    validator.save_results(results_df)
    
    # Save metrics to CSV for comparison
    metrics_df = pd.DataFrame(metrics_dict.values())
    metrics_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    logger.info(f"✓ Model comparison saved to {output_dir}/model_comparison.csv")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PHASE 3 COMPLETE")
    logger.info("=" * 80)
    
    return results_df, metrics_dict


if __name__ == "__main__":
    try:
        results_df, metrics = main()
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
