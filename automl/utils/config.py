"""
Configuration globale du projet AutoML.
"""
import os


class Config:
    """
    Configuration globale du projet AutoML.

    Cette classe contient tous les paramètres par défaut utilisés
    dans l'ensemble du projet. Les valeurs peuvent être modifiées
    selon les besoins.
    """

    # ========== CHEMINS ==========
    # Chemin vers les données sur le cluster Skinner
    DATA_PATH = "/info/corpus/ChallengeMachineLearning"

    # Répertoire de sauvegarde des modèles entraînés
    MODELS_SAVE_PATH = "./saved_models"

    # Répertoire de sauvegarde des résultats d'évaluation
    RESULTS_PATH = "./results"

    # ========== SPLITS DES DONNÉES ==========
    # Proportion pour l'ensemble d'entraînement
    TRAIN_SIZE = 0.7

    # Proportion pour l'ensemble de validation
    VALID_SIZE = 0.15

    # Proportion pour l'ensemble de test
    TEST_SIZE = 0.15

    # ========== PRÉPROCESSING ==========
    # Stratégie pour gérer les valeurs manquantes
    # Options: 'mean', 'median', 'most_frequent', 'drop'
    HANDLE_MISSING = 'mean'

    # Normaliser les features numériques (StandardScaler)
    SCALE_FEATURES = True

    # Encoder les variables catégorielles
    ENCODE_CATEGORICAL = True

    # ========== MODÈLES ==========
    # Liste des modèles à entraîner
    # Options possibles: 'logistic_regression', 'random_forest', 'svm',
    #                   'gradient_boosting', 'knn', 'decision_tree'
    MODELS_TO_TRAIN = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting'
    ]

    # ========== OPTIMISATION ==========
    # Nombre d'itérations pour la recherche d'hyperparamètres
    N_ITER_SEARCH = 20

    # Nombre de folds pour la validation croisée
    CV_FOLDS = 5

    # Métrique d'optimisation
    # Classification: 'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    # Régression: 'neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'
    OPTIMIZATION_METRIC = 'accuracy'  # Sera adapté selon task_type

    # ========== ÉVALUATION ==========
    # Métriques à calculer pour la classification
    CLASSIFICATION_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'roc_auc'
    ]

    # Métriques à calculer pour la régression
    REGRESSION_METRICS = [
        'r2',
        'mse',
        'rmse',
        'mae'
    ]

    # ========== GÉNÉRAL ==========
    # Graine aléatoire pour reproductibilité
    RANDOM_STATE = 42

    # Afficher les informations de progression
    VERBOSE = True

    # Nombre de jobs parallèles (-1 = tous les CPU)
    N_JOBS = -1

    @classmethod
    def create_directories(cls):
        """
        Crée les répertoires nécessaires s'ils n'existent pas.
        """
        os.makedirs(cls.MODELS_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.RESULTS_PATH, exist_ok=True)

    @classmethod
    def display(cls):
        """
        Affiche la configuration actuelle.
        """
        print("=" * 50)
        print("CONFIGURATION AUTOML")
        print("=" * 50)
        print(f"DATA_PATH: {cls.DATA_PATH}")
        print(f"MODELS_SAVE_PATH: {cls.MODELS_SAVE_PATH}")
        print(f"RESULTS_PATH: {cls.RESULTS_PATH}")
        print(f"\nSplits: Train={cls.TRAIN_SIZE}, Valid={cls.VALID_SIZE}, "
              f"Test={cls.TEST_SIZE}")
        print(f"\nPreprocessing:")
        print(f"  - Handle missing: {cls.HANDLE_MISSING}")
        print(f"  - Scale features: {cls.SCALE_FEATURES}")
        print(f"  - Encode categorical: {cls.ENCODE_CATEGORICAL}")
        print(f"\nRandom state: {cls.RANDOM_STATE}")
        print(f"Verbose: {cls.VERBOSE}")
        print("=" * 50)
