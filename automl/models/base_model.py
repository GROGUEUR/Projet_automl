"""
Module contenant la classe de base pour tous les modèles AutoML.

Cette classe encapsule un modèle sklearn avec des métadonnées et
des méthodes standardisées pour l'entraînement, la prédiction et
la sauvegarde.
"""
from abc import ABC
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, r2_score
import joblib
from datetime import datetime
import time
import os
from typing import Optional
import numpy as np


class BaseModel:
    """
    Classe de base pour tous les modèles AutoML.
    Encapsule un modèle sklearn avec métadonnées.
    """

    def __init__(self, name: str, model: BaseEstimator, task_type: str):
        """
        Initialise un modèle de base.

        Args:
            name: nom du modèle (ex: "RandomForest")
            model: instance d'un modèle sklearn
            task_type: 'classification' ou 'regression'
        """
        self.name = name
        self.model = model
        self.task_type = task_type
        self.is_fitted = False
        self.train_score = None
        self.valid_score = None
        self.training_time = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Entraîne le modèle et calcule les scores.

        Args:
            X_train, y_train: données d'entraînement
            X_valid, y_valid: données de validation (optionnel)

        Returns:
            self: permet le chaînage de méthodes
        """
        start_time = time.time()

        # Entraînement du modèle
        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Calcul du temps d'entraînement
        self.training_time = time.time() - start_time

        # Calcul des scores
        if self.task_type == 'classification':
            self.train_score = accuracy_score(y_train, self.model.predict(X_train))
            if X_valid is not None and y_valid is not None:
                self.valid_score = accuracy_score(y_valid, self.model.predict(X_valid))

        elif self.task_type == 'regression':
            self.train_score = r2_score(y_train, self.model.predict(X_train))
            if X_valid is not None and y_valid is not None:
                self.valid_score = r2_score(y_valid, self.model.predict(X_valid))

        return self

    def predict(self, X):
        """
        Prédit sur de nouvelles données.

        Args:
            X: données à prédire

        Returns:
            Prédictions

        Raises:
            RuntimeError: si le modèle n'est pas entraîné
        """
        if not self.is_fitted:
            raise RuntimeError(f"Le modèle {self.name} n'est pas entraîné. Appelez fit() d'abord.")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Prédit les probabilités (classification uniquement).

        Args:
            X: données à prédire

        Returns:
            Probabilités pour chaque classe

        Raises:
            RuntimeError: si le modèle n'est pas entraîné
            AttributeError: si le modèle ne supporte pas predict_proba
        """
        if not self.is_fitted:
            raise RuntimeError(f"Le modèle {self.name} n'est pas entraîné. Appelez fit() d'abord.")

        if self.task_type != 'classification':
            raise ValueError("predict_proba n'est disponible que pour la classification")

        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"Le modèle {self.name} ne supporte pas predict_proba")

        return self.model.predict_proba(X)

    def save(self, path: str):
        """
        Sauvegarde le modèle avec joblib.
        Format: {path}/{name}_{timestamp}.joblib

        Args:
            path: répertoire de sauvegarde

        Returns:
            str: chemin complet du fichier sauvegardé
        """
        if not self.is_fitted:
            raise RuntimeError(f"Le modèle {self.name} n'est pas entraîné. Rien à sauvegarder.")

        # Créer le répertoire s'il n'existe pas
        os.makedirs(path, exist_ok=True)

        # Générer un nom de fichier avec timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name}_{timestamp}.joblib"
        filepath = os.path.join(path, filename)

        # Sauvegarder l'objet complet
        joblib.dump(self, filepath)

        return filepath

    @staticmethod
    def load(path: str):
        """
        Charge un modèle sauvegardé.

        Args:
            path: chemin complet vers le fichier .joblib

        Returns:
            BaseModel: modèle chargé
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")

        model = joblib.load(path)

        if not isinstance(model, BaseModel):
            raise ValueError(f"Le fichier {path} ne contient pas un objet BaseModel valide")

        return model

    def get_params(self):
        """
        Retourne les hyperparamètres du modèle sklearn.

        Returns:
            dict: dictionnaire des hyperparamètres
        """
        return self.model.get_params()

    def set_params(self, **params):
        """
        Modifie les hyperparamètres.

        Args:
            **params: nouveaux hyperparamètres

        Returns:
            self: permet le chaînage de méthodes
        """
        self.model.set_params(**params)
        # Réinitialiser l'état d'entraînement
        self.is_fitted = False
        self.train_score = None
        self.valid_score = None
        self.training_time = None
        return self

    def __repr__(self):
        """Représentation string du modèle."""
        status = "fitted" if self.is_fitted else "not fitted"
        score_info = f", valid_score={self.valid_score:.4f}" if self.valid_score is not None else ""
        return f"BaseModel(name='{self.name}', task_type='{self.task_type}', {status}{score_info})"
