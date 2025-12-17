"""
src/model_trainer.py
Train 4 models with proper class imbalance handling.

PHASE 2 IMPROVEMENTS:
- LogisticRegression: class_weight='balanced' (FIX for Priority 4)
- RandomForest: class_weight='balanced'
- XGBoost: scale_pos_weight parameter
- LSTM: SMOTE oversampling

This is the corrected Priority 4 implementation.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_logistic_regression_balanced(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparams: Dict = None
) -> LogisticRegression:
    """
    Train LogisticRegression with class_weight='balanced'.
    
    FIX: Addresses Priority 4 bug where LogReg predicted only class 0.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparams: Optional hyperparameters for GridSearch
    
    Returns:
        Trained LogisticRegression model
    """
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    logger.info(f"Training set class distribution: {class_dist}")
    
    imbalance_ratio = max(counts) / min(counts)
    logger.warning(f"⚠ Class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 10:
        logger.warning("✗ SEVERE class imbalance detected!")
        logger.info("✓ USING: class_weight='balanced' to handle imbalance")
    
    # Default parameters with class_weight='balanced'
    params = {
        'class_weight': 'balanced',  # CRITICAL FIX
        'max_iter': 1000,
        'random_state': 42,
        'solver': 'lbfgs'
    }
    
    # Optional hyperparameter tuning
    if hyperparams:
        logger.info("Performing GridSearchCV for hyperparameter tuning...")
        param_grid = {
            'C': hyperparams.get('C', [0.001, 0.01, 0.1, 1, 10]),
            'penalty': hyperparams.get('penalty', ['l2']),
        }
        
        base_model = LogisticRegression(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='recall', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✓ Best parameters: {grid_search.best_params_}")
        logger.info(f"✓ Best CV recall: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    # Train with balanced class weight
    try:
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        logger.info("✓ LogisticRegression training complete")
        
        # Check predictions on training set
        train_pred = model.predict(X_train)
        train_pred_proba = model.predict_proba(X_train)
        
        logger.info(f"  Training predictions: {np.unique(train_pred, return_counts=True)}")
        logger.info(f"  Sample probabilities: {train_pred_proba[:5]}")
        
        return model
    except Exception as e:
        logger.error(f"✗ LogisticRegression training failed: {e}")
        raise


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparams: Dict = None
) -> RandomForestClassifier:
    """
    Train RandomForest with class_weight='balanced'.
    
    Better for nonlinear patterns and naturally handles class imbalance better.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparams: Optional hyperparameters for GridSearch
    
    Returns:
        Trained RandomForestClassifier
    """
    logger.info("Training RandomForest classifier...")
    
    # Check class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    imbalance_ratio = max(counts) / min(counts)
    logger.warning(f"⚠ Class imbalance ratio: {imbalance_ratio:.1f}:1")
    
    params = {
        'n_estimators': 200,
        'class_weight': 'balanced',
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if hyperparams:
        logger.info("Performing GridSearchCV for RandomForest...")
        param_grid = {
            'n_estimators': hyperparams.get('n_estimators', [100, 200, 300]),
            'max_depth': hyperparams.get('max_depth', [10, 15, 20]),
            'min_samples_split': hyperparams.get('min_samples_split', [5, 10]),
        }
        
        base_model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring='recall', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f"✓ Best parameters: {grid_search.best_params_}")
        logger.info(f"✓ Best CV recall: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    
    try:
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("✓ RandomForest training complete")
        return model
    except Exception as e:
        logger.error(f"✗ RandomForest training failed: {e}")
        raise


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hyperparams: Dict = None
) -> xgb.XGBClassifier:
    """
    Train XGBoost with scale_pos_weight for class imbalance.
    
    XGBoost is very effective for nonlinear relationships.
    
    Args:
        X_train: Training features
        y_train: Training targets
        hyperparams: Optional hyperparameters
    
    Returns:
        Trained XGBClassifier
    """
    logger.info("Training XGBoost classifier...")
    
    # Calculate scale_pos_weight for imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    logger.info(f"  Negative samples: {neg_count}, Positive samples: {pos_count}")
    logger.info(f"  scale_pos_weight: {scale_pos_weight:.2f}")
    
    params = {
        'scale_pos_weight': scale_pos_weight,
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    try:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, verbose=False)
        logger.info("✓ XGBoost training complete")
        return model
    except Exception as e:
        logger.error(f"✗ XGBoost training failed: {e}")
        raise


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sequence_length: int = 6,
    use_smote: bool = True
) -> Sequential:
    """
    Train LSTM with sequence reshaping and optional SMOTE.
    
    Args:
        X_train: Training features (samples, features)
        y_train: Training targets
        sequence_length: Length of sequences for LSTM
        use_smote: Whether to apply SMOTE oversampling
    
    Returns:
        Trained LSTM model
    """
    logger.info("Training LSTM model...")
    
    # Apply SMOTE if requested
    if use_smote:
        logger.info("Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"✓ After SMOTE: {np.sum(y_train == 1)} positive, {np.sum(y_train == 0)} negative")
    
    # Reshape for LSTM (samples, sequence_length, features)
    n_samples, n_features = X_train.shape
    
    # Create sequences
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_train) - sequence_length):
        X_sequences.append(X_train[i:i+sequence_length])
        y_sequences.append(y_train[i+sequence_length])
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    logger.info(f"✓ Sequence shape: {X_sequences.shape}")
    logger.info(f"✓ Target shape: {y_sequences.shape}")
    
    # Build LSTM model
    model = Sequential([
        LSTM(64, activation='relu', input_shape=(sequence_length, n_features)),
        Dropout(0.3),
        LSTM(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train with early stopping
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    
    try:
        history = model.fit(
            X_sequences, y_sequences,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )
        logger.info(f"✓ LSTM training complete (stopped at epoch {len(history.history['loss'])})")
        return model
    except Exception as e:
        logger.error(f"✗ LSTM training failed: {e}")
        raise


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    output_dir: str = 'models/',
    hyperparameter_tuning: bool = False
) -> Dict:
    """
    Train all 4 models with proper class imbalance handling.
    
    This is the corrected Priority 4 implementation.
    
    Args:
        X_train: Training features (should be 57 features from data_loader)
        y_train: Training targets
        output_dir: Directory to save models
        hyperparameter_tuning: Whether to perform GridSearchCV
    
    Returns:
        Dict of trained models
    """
    logger.info("=" * 80)
    logger.info("PRIORITY 4 (CORRECTED): Training All Models")
    logger.info("=" * 80)
    
    Path(output_dir).mkdir(exist_ok=True)
    models = {}
    
    # 1. Logistic Regression (FIXED)
    logger.info("\n[1/4] Training LogisticRegression with balanced class weight...")
    try:
        models['logreg'] = train_logistic_regression_balanced(
            X_train, y_train,
            hyperparams={'C': [0.001, 0.01, 0.1, 1, 10]} if hyperparameter_tuning else None
        )
        with open(f'{output_dir}/logreg_balanced_model.pkl', 'wb') as f:
            pickle.dump(models['logreg'], f)
        logger.info("✓ Saved: logreg_balanced_model.pkl")
    except Exception as e:
        logger.error(f"✗ LogReg training failed: {e}")
    
    # 2. Random Forest
    logger.info("\n[2/4] Training RandomForest...")
    try:
        models['rf'] = train_random_forest(
            X_train, y_train,
            hyperparams=None if not hyperparameter_tuning else {}
        )
        with open(f'{output_dir}/randomforest_model.pkl', 'wb') as f:
            pickle.dump(models['rf'], f)
        logger.info("✓ Saved: randomforest_model.pkl")
    except Exception as e:
        logger.error(f"✗ RandomForest training failed: {e}")
    
    # 3. XGBoost
    logger.info("\n[3/4] Training XGBoost...")
    try:
        models['xgb'] = train_xgboost(X_train, y_train)
        with open(f'{output_dir}/xgboost_model.pkl', 'wb') as f:
            pickle.dump(models['xgb'], f)
        logger.info("✓ Saved: xgboost_model.pkl")
    except Exception as e:
        logger.error(f"✗ XGBoost training failed: {e}")
    
    # 4. LSTM
    logger.info("\n[4/4] Training LSTM with SMOTE...")
    try:
        models['lstm'] = train_lstm(X_train, y_train, use_smote=True)
        models['lstm'].save(f'{output_dir}/lstm_model.h5')
        logger.info("✓ Saved: lstm_model.h5")
    except Exception as e:
        logger.error(f"✗ LSTM training failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✓ All models trained. Total: {len(models)} models saved to {output_dir}")
    logger.info("=" * 80)
    
    return models


if __name__ == "__main__":
    # Example usage
    from data_loader import load_engineered_features
    
    try:
        df, X, y = load_engineered_features('data/processed/features_engineered_monthly.csv')
        
        # Use first 80% for training (walk-forward would split differently)
        train_size = int(0.8 * len(X))
        X_train, y_train = X[:train_size], y[:train_size]
        
        logger.info(f"Training on {len(X_train)} samples...")
        models = train_all_models(X_train, y_train, hyperparameter_tuning=False)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
