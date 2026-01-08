"""
AutoML - Système d'apprentissage automatique automatisé.

Ce paquet fournit une interface simple pour entraîner et évaluer
automatiquement des modèles de machine learning.

Usage:
    >>> import automl
    >>> automl.fit(data_path="/path/to/data")
    >>> automl.eval()
"""
from .core import fit, eval, get_data, reset
from .models import get_trained_models, get_best_model

__version__ = '0.1.0'
__all__ = ['fit', 'eval', 'get_data', 'reset', 'get_trained_models', 'get_best_model']
