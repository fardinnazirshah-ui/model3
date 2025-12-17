"""
src/__init__.py
Recession Predictor Package
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "Machine learning framework for recession prediction using walk-forward validation"

from .data_loader import load_fred_data, load_worldbank_data, load_nber_data
from .feature_engineer import engineer_features
from .model_trainer import train_models
from .validator import walk_forward_validation
from .utils import calculate_metrics, plot_confusion_matrix

__all__ = [
    'load_fred_data',
    'load_worldbank_data',
    'load_nber_data',
    'engineer_features',
    'train_models',
    'walk_forward_validation',
    'calculate_metrics',
    'plot_confusion_matrix'
]
