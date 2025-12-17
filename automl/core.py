"""
Module principal de l'interface AutoML.

Ce module fournit les fonctions principales fit(), eval() et get_data()
qui constituent l'interface utilisateur du système AutoML.
"""
from typing import Optional, Dict, Any
import numpy as np

from .data.loader import DataLoader
from .data.preprocessing import DataPreprocessor, train_valid_test_split
from .utils.config import Config


# ========== Variables globales pour stocker l'état ==========
_data_loader: Optional[DataLoader] = None
_preprocessor: Optional[DataPreprocessor] = None
_X_train: Optional[np.ndarray] = None
_X_valid: Optional[np.ndarray] = None
_X_test: Optional[np.ndarray] = None
_y_train: Optional[np.ndarray] = None
_y_valid: Optional[np.ndarray] = None
_y_test: Optional[np.ndarray] = None
_task_type: Optional[str] = None
_trained_models: Dict[str, Any] = {}  # Sera utilisé par Personne 2


def fit(data_path: str, **kwargs) -> bool:
    """
    Charge les données, les prétraite et lance l'entraînement.
    Point d'entrée principal du système AutoML.

    Cette fonction:
    1. Charge les données depuis le chemin spécifié
    2. Détecte automatiquement le type de tâche (classification/régression)
    3. Prétraite les données (gestion des manquantes, normalisation, encodage)
    4. Sépare les données en ensembles train/valid/test
    5. Lance l'entraînement des modèles (module de Personne 2)

    Args:
        data_path: Chemin vers le répertoire ou fichier de données
        **kwargs: Arguments supplémentaires pour personnaliser:
            - train_size (float): Proportion pour l'entraînement (défaut: 0.7)
            - valid_size (float): Proportion pour la validation (défaut: 0.15)
            - test_size (float): Proportion pour le test (défaut: 0.15)
            - handle_missing (str): Stratégie pour valeurs manquantes (défaut: 'mean')
            - scale (bool): Normaliser les features (défaut: True)
            - encode_categorical (bool): Encoder les catégorielles (défaut: True)
            - random_state (int): Graine aléatoire (défaut: 42)
            - verbose (bool): Afficher les informations (défaut: True)

    Returns:
        bool: True si l'exécution s'est bien déroulée

    Raises:
        FileNotFoundError: Si le chemin de données n'existe pas
        ValueError: Si les données sont invalides ou mal formatées

    Example:
        >>> import automl
        >>> automl.fit(data_path="/path/to/data")
        >>> automl.eval()
    """
    global _data_loader, _preprocessor, _X_train, _X_valid, _X_test
    global _y_train, _y_valid, _y_test, _task_type, _trained_models

    # Récupérer les paramètres avec valeurs par défaut
    train_size = kwargs.get('train_size', Config.TRAIN_SIZE)
    valid_size = kwargs.get('valid_size', Config.VALID_SIZE)
    test_size = kwargs.get('test_size', Config.TEST_SIZE)
    handle_missing = kwargs.get('handle_missing', Config.HANDLE_MISSING)
    scale = kwargs.get('scale', Config.SCALE_FEATURES)
    encode_categorical = kwargs.get('encode_categorical', Config.ENCODE_CATEGORICAL)
    random_state = kwargs.get('random_state', Config.RANDOM_STATE)
    verbose = kwargs.get('verbose', Config.VERBOSE)

    if verbose:
        print("=" * 60)
        print("AUTOML - SYSTÈME D'APPRENTISSAGE AUTOMATIQUE")
        print("=" * 60)
        print()

    # ========== ÉTAPE 1: Chargement des données ==========
    if verbose:
        print("ÉTAPE 1/4: Chargement des données...")
        print(f"  Source: {data_path}")

    try:
        _data_loader = DataLoader(data_path)
        X, y, _task_type = _data_loader.load_data()

        if verbose:
            info = _data_loader.get_info()
            print(f"  ✓ Données chargées avec succès!")
            print(f"  - Nombre d'échantillons: {info['n_samples']}")
            print(f"  - Nombre de features: {info['n_features']}")
            print(f"  - Type de tâche: {_task_type}")
            if _task_type == 'classification':
                print(f"  - Nombre de classes: {info['n_classes']}")
            if info['missing_values'] > 0:
                print(f"  - Valeurs manquantes: {info['missing_values']}")
            print()

    except Exception as e:
        print(f"  ✗ Erreur lors du chargement des données: {e}")
        raise

    # ========== ÉTAPE 2: Prétraitement des données ==========
    if verbose:
        print("ÉTAPE 2/4: Prétraitement des données...")
        print(f"  - Gestion des valeurs manquantes: {handle_missing}")
        print(f"  - Normalisation: {scale}")
        print(f"  - Encodage catégoriel: {encode_categorical}")

    try:
        _preprocessor = DataPreprocessor(
            handle_missing=handle_missing,
            scale=scale,
            encode_categorical=encode_categorical
        )

        # Fit et transform sur toutes les données
        X_processed = _preprocessor.fit_transform(X)

        if verbose:
            print(f"  ✓ Prétraitement effectué avec succès!")
            print(f"  - Dimensions après traitement: {X_processed.shape}")
            print()

    except Exception as e:
        print(f"  ✗ Erreur lors du prétraitement: {e}")
        raise

    # ========== ÉTAPE 3: Séparation des données ==========
    if verbose:
        print("ÉTAPE 3/4: Séparation des données...")
        print(f"  - Train: {train_size*100:.0f}%")
        print(f"  - Validation: {valid_size*100:.0f}%")
        print(f"  - Test: {test_size*100:.0f}%")

    try:
        splits = train_valid_test_split(
            X_processed, y,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
            task_type=_task_type
        )

        _X_train = splits['X_train']
        _X_valid = splits['X_valid']
        _X_test = splits['X_test']
        _y_train = splits['y_train']
        _y_valid = splits['y_valid']
        _y_test = splits['y_test']

        if verbose:
            print(f"  ✓ Données séparées avec succès!")
            print(f"  - Train: {_X_train.shape[0]} échantillons")
            print(f"  - Validation: {_X_valid.shape[0]} échantillons")
            print(f"  - Test: {_X_test.shape[0]} échantillons")
            print()

    except Exception as e:
        print(f"  ✗ Erreur lors de la séparation des données: {e}")
        raise

    # ========== ÉTAPE 4: Entraînement des modèles ==========
    if verbose:
        print("ÉTAPE 4/4: Entraînement des modèles...")

    try:
        # Import du module models (sera créé par Personne 2)
        # Pour l'instant, on met un placeholder
        try:
            from .models import train_models
            _trained_models = train_models(
                _X_train, _y_train,
                _X_valid, _y_valid,
                _task_type,
                verbose=verbose
            )
            if verbose:
                print(f"  ✓ {len(_trained_models)} modèles entraînés avec succès!")
        except ImportError:
            if verbose:
                print("  ⚠  Module 'models' non disponible (sera implémenté par Personne 2)")
                print("  → Les données sont prêtes pour l'entraînement!")

        if verbose:
            print()
            print("=" * 60)
            print("SUCCÈS: Pipeline de données complété!")
            print("=" * 60)
            print()

    except Exception as e:
        if verbose:
            print(f"  ⚠  Avertissement lors de l'entraînement: {e}")
        # Ne pas lever l'exception pour permettre aux autres modules de se connecter

    return True


def eval(**kwargs) -> Dict[str, Any]:
    """
    Évalue les modèles entraînés sur l'ensemble de test.

    Cette fonction appelle le module d'évaluation (Personne 4) pour
    calculer les métriques de performance sur l'ensemble de test.

    Args:
        **kwargs: Arguments supplémentaires pour l'évaluation:
            - verbose (bool): Afficher les résultats (défaut: True)
            - save_results (bool): Sauvegarder les résultats (défaut: True)

    Returns:
        Dict[str, Any]: Dictionnaire contenant les résultats d'évaluation

    Raises:
        RuntimeError: Si fit() n'a pas été appelé avant eval()

    Example:
        >>> import automl
        >>> automl.fit(data_path="/path/to/data")
        >>> results = automl.eval()
    """
    global _X_test, _y_test, _task_type, _trained_models

    # Vérifier que fit() a été appelé
    if _X_test is None or _y_test is None:
        raise RuntimeError(
            "Les données de test ne sont pas disponibles. "
            "Appelez d'abord fit() pour charger et préparer les données."
        )

    if len(_trained_models) == 0:
        print("⚠  Aucun modèle entraîné à évaluer.")
        print("   Le module 'models' doit être implémenté par Personne 2.")
        return {}

    verbose = kwargs.get('verbose', Config.VERBOSE)

    if verbose:
        print("=" * 60)
        print("ÉVALUATION DES MODÈLES")
        print("=" * 60)
        print()

    try:
        # Import du module evaluation (sera créé par Personne 4)
        from .evaluation import evaluate_models

        results = evaluate_models(
            _trained_models,
            _X_test, _y_test,
            _task_type,
            **kwargs
        )

        return results

    except ImportError:
        if verbose:
            print("⚠  Module 'evaluation' non disponible (sera implémenté par Personne 4)")
            print("   Les données de test sont prêtes pour l'évaluation!")
        return {}


def get_data() -> Dict[str, Any]:
    """
    Retourne les données actuellement chargées.

    Utile pour le debugging ou pour accéder aux données depuis
    d'autres modules du système.

    Returns:
        Dict[str, Any]: Dictionnaire contenant:
            - X_train, X_valid, X_test: Features des différents ensembles
            - y_train, y_valid, y_test: Targets des différents ensembles
            - task_type: Type de tâche ('classification' ou 'regression')
            - trained_models: Dictionnaire des modèles entraînés

    Example:
        >>> data = automl.get_data()
        >>> X_train = data['X_train']
        >>> y_train = data['y_train']
    """
    return {
        'X_train': _X_train,
        'X_valid': _X_valid,
        'X_test': _X_test,
        'y_train': _y_train,
        'y_valid': _y_valid,
        'y_test': _y_test,
        'task_type': _task_type,
        'trained_models': _trained_models
    }


def reset() -> None:
    """
    Réinitialise l'état global du système.

    Utile pour nettoyer la mémoire ou recommencer avec de nouvelles données.
    """
    global _data_loader, _preprocessor, _X_train, _X_valid, _X_test
    global _y_train, _y_valid, _y_test, _task_type, _trained_models

    _data_loader = None
    _preprocessor = None
    _X_train = None
    _X_valid = None
    _X_test = None
    _y_train = None
    _y_valid = None
    _y_test = None
    _task_type = None
    _trained_models = {}

    print("✓ État du système réinitialisé")
