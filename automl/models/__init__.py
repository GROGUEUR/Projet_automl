"""
Module d'entraînement des modèles pour AutoML.

Ce module fournit toutes les fonctionnalités nécessaires pour :
- Créer différents types de modèles sklearn
- Entraîner et comparer plusieurs modèles
- Sélectionner le meilleur modèle selon différents critères
- Sauvegarder et charger les modèles

Implémenté par Personne 2.
"""
from .base_model import BaseModel
from .model_factory import ModelFactory
from .model_trainer import ModelTrainer
from .model_selector import ModelSelector


# Variable globale pour stocker le trainer (pour l'intégration avec core.py)
_model_trainer = None


def train_models(X_train, y_train, X_valid, y_valid, task_type, **kwargs):
    """
    Entraîne tous les modèles disponibles.
    Appelée par fit() après le chargement des données.

    Cette fonction est le point d'entrée principal du module models,
    appelé depuis automl.core.fit().

    Args:
        X_train, y_train: données d'entraînement
        X_valid, y_valid: données de validation
        task_type: 'classification' ou 'regression'
        **kwargs: arguments supplémentaires:
            - verbose (bool): afficher les logs
            - random_state (int): seed
            - models (List[BaseModel]): modèles personnalisés (optionnel)

    Returns:
        Dict[str, BaseModel]: dictionnaire nom -> modèle entraîné
    """
    global _model_trainer

    verbose = kwargs.get('verbose', True)
    random_state = kwargs.get('random_state', 42)
    models = kwargs.get('models', None)

    # Créer le trainer
    _model_trainer = ModelTrainer(
        task_type=task_type,
        models=models,
        random_state=random_state,
        verbose=verbose
    )

    # Entraîner tous les modèles
    results = _model_trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Sélectionner le meilleur
    if results:
        best_model = _model_trainer.select_best_model(metric='valid_score')

        if verbose:
            print("\n📊 Résumé des performances:")
            summary = _model_trainer.get_results_summary()
            print(summary.to_string(index=False))
            print()

    # Retourner le dictionnaire des modèles entraînés
    return _model_trainer.get_all_models()


def get_trained_models():
    """
    Retourne le trainer avec tous les modèles (pour accès par autres modules).

    Returns:
        ModelTrainer: instance du trainer avec tous les modèles entraînés

    Raises:
        RuntimeError: si aucun modèle n'a été entraîné
    """
    if _model_trainer is None:
        raise RuntimeError(
            "Aucun modèle entraîné. Appelez automl.fit() d'abord."
        )
    return _model_trainer


def get_best_model():
    """
    Retourne le meilleur modèle sélectionné.

    Returns:
        BaseModel: meilleur modèle

    Raises:
        RuntimeError: si aucun modèle n'a été entraîné ou sélectionné
    """
    if _model_trainer is None:
        raise RuntimeError(
            "Aucun modèle entraîné. Appelez automl.fit() d'abord."
        )

    if _model_trainer.best_model is None:
        raise RuntimeError(
            "Aucun meilleur modèle sélectionné. Le processus d'entraînement "
            "n'a peut-être pas abouti correctement."
        )

    return _model_trainer.best_model


def get_model(name: str):
    """
    Récupère un modèle spécifique par son nom.

    Args:
        name: nom du modèle

    Returns:
        BaseModel: modèle demandé

    Raises:
        RuntimeError: si aucun modèle n'a été entraîné
        KeyError: si le modèle n'existe pas
    """
    if _model_trainer is None:
        raise RuntimeError(
            "Aucun modèle entraîné. Appelez automl.fit() d'abord."
        )

    return _model_trainer.get_model(name)


def save_models(save_dir: str, best_only: bool = False):
    """
    Sauvegarde les modèles entraînés.

    Args:
        save_dir: répertoire de sauvegarde
        best_only: sauvegarder uniquement le meilleur modèle

    Returns:
        List[str]: chemins des fichiers sauvegardés

    Raises:
        RuntimeError: si aucun modèle n'a été entraîné
    """
    if _model_trainer is None:
        raise RuntimeError(
            "Aucun modèle entraîné. Appelez automl.fit() d'abord."
        )

    if best_only:
        path = _model_trainer.save_best_model(save_dir)
        return [path]
    else:
        return _model_trainer.save_all_models(save_dir)


def reset_models():
    """
    Réinitialise l'état du module (pour debugging ou réentraînement).
    """
    global _model_trainer
    _model_trainer = None


# Exporter tous les éléments publics
__all__ = [
    # Classes
    'BaseModel',
    'ModelFactory',
    'ModelTrainer',
    'ModelSelector',
    # Fonctions d'intégration
    'train_models',
    'get_trained_models',
    'get_best_model',
    'get_model',
    'save_models',
    'reset_models',
]
