"""
Module de prétraitement des données pour AutoML.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime


class DataPreprocessor:
    """
    Classe pour prétraiter les données.

    Gère le traitement des valeurs manquantes, la normalisation des features
    numériques et l'encodage des variables catégorielles.
    """

    def __init__(self, handle_missing: str = 'mean', scale: bool = True,
                 encode_categorical: bool = True):
        """
        Initialise le DataPreprocessor.

        Args:
            handle_missing: stratégie pour valeurs manquantes
                          ('mean', 'median', 'most_frequent', 'drop')
            scale: si True, normalise les features numériques
            encode_categorical: si True, encode les variables catégorielles

        Raises:
            ValueError: Si handle_missing n'est pas une stratégie valide
        """
        valid_strategies = ['mean', 'median', 'most_frequent', 'drop']
        if handle_missing not in valid_strategies:
            raise ValueError(
                f"handle_missing doit être dans {valid_strategies}, "
                f"reçu: {handle_missing}"
            )

        self.handle_missing = handle_missing
        self.scale = scale
        self.encode_categorical = encode_categorical

        # Transformers pour features numériques
        self.numeric_imputer = None
        self.scaler = None

        # Transformers pour features catégorielles
        self.categorical_imputer = None
        self.encoders = {}

        # Informations sur les colonnes
        self.numeric_columns = None
        self.categorical_columns = None
        self.feature_names = None

        # Fitted state
        self.is_fitted = False

    def _identify_column_types(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple:
        """
        Identifie les colonnes numériques et catégorielles.

        Args:
            X: Données d'entrée

        Returns:
            Tuple contenant (numeric_columns, categorical_columns)
        """
        # Convertir en DataFrame si nécessaire
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        numeric_cols = []
        categorical_cols = []

        for col in X_df.columns:
            # Si le type est object ou category, c'est catégoriel
            if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                categorical_cols.append(col)
            else:
                # Sinon c'est numérique
                numeric_cols.append(col)

        return numeric_cols, categorical_cols

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Optional[Union[np.ndarray, pd.Series]] = None):
        """
        Apprend les transformations sur les données d'entraînement.

        Args:
            X: Features d'entraînement
            y: Target (optionnel, non utilisé pour l'instant)

        Returns:
            self: Retourne l'instance pour chaînage
        """
        # Convertir en DataFrame si nécessaire pour faciliter le traitement
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        # Identifier les types de colonnes
        self.numeric_columns, self.categorical_columns = \
            self._identify_column_types(X_df)

        # Traiter les features numériques
        if len(self.numeric_columns) > 0:
            X_numeric = X_df[self.numeric_columns]

            # Imputation des valeurs manquantes
            if self.handle_missing != 'drop':
                self.numeric_imputer = SimpleImputer(strategy=self.handle_missing)
                self.numeric_imputer.fit(X_numeric)

            # Normalisation
            if self.scale:
                # Fit sur les données imputées si nécessaire
                if self.numeric_imputer is not None:
                    X_numeric_imputed = self.numeric_imputer.transform(X_numeric)
                else:
                    X_numeric_imputed = X_numeric.values

                self.scaler = StandardScaler()
                self.scaler.fit(X_numeric_imputed)

        # Traiter les features catégorielles
        if len(self.categorical_columns) > 0 and self.encode_categorical:
            X_categorical = X_df[self.categorical_columns]

            # Imputation des valeurs manquantes (most_frequent pour catégoriel)
            if self.handle_missing != 'drop':
                self.categorical_imputer = SimpleImputer(strategy='most_frequent')
                self.categorical_imputer.fit(X_categorical)

            # Encodage avec LabelEncoder pour chaque colonne
            X_cat_imputed = X_categorical.copy()
            if self.categorical_imputer is not None:
                X_cat_imputed = pd.DataFrame(
                    self.categorical_imputer.transform(X_categorical),
                    columns=X_categorical.columns
                )

            for col in self.categorical_columns:
                encoder = LabelEncoder()
                encoder.fit(X_cat_imputed[col].astype(str))
                self.encoders[col] = encoder

        self.is_fitted = True
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Applique les transformations apprises.

        Args:
            X: Données à transformer

        Returns:
            X_transformed: Données transformées (numpy array)

        Raises:
            RuntimeError: Si transform est appelé avant fit
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Le preprocessor doit être fitted avant de transformer. "
                "Appelez fit() ou fit_transform() d'abord."
            )

        # Convertir en DataFrame si nécessaire
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X)
        else:
            X_df = X.copy()

        transformed_parts = []

        # Transformer les features numériques
        if len(self.numeric_columns) > 0:
            X_numeric = X_df[self.numeric_columns]

            # Imputation
            if self.numeric_imputer is not None:
                X_numeric = self.numeric_imputer.transform(X_numeric)
            else:
                X_numeric = X_numeric.values

            # Normalisation
            if self.scaler is not None:
                X_numeric = self.scaler.transform(X_numeric)

            transformed_parts.append(X_numeric)

        # Transformer les features catégorielles
        if len(self.categorical_columns) > 0 and self.encode_categorical:
            X_categorical = X_df[self.categorical_columns]

            # Imputation
            if self.categorical_imputer is not None:
                X_categorical = pd.DataFrame(
                    self.categorical_imputer.transform(X_categorical),
                    columns=X_categorical.columns
                )

            # Encodage
            encoded_cols = []
            for col in self.categorical_columns:
                encoder = self.encoders[col]
                # Gérer les catégories inconnues en les remplaçant par la plus fréquente
                col_values = X_categorical[col].astype(str)
                # Remplacer les valeurs inconnues par la première classe connue
                known_classes = set(encoder.classes_)
                col_values = col_values.apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                encoded = encoder.transform(col_values)
                encoded_cols.append(encoded)

            X_categorical_encoded = np.column_stack(encoded_cols)
            transformed_parts.append(X_categorical_encoded)

        # Combiner toutes les parties
        if len(transformed_parts) == 0:
            return X_df.values

        X_transformed = np.hstack(transformed_parts)
        return X_transformed

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame],
                      y: Optional[Union[np.ndarray, pd.Series]] = None) -> np.ndarray:
        """
        Fit puis transform en une seule étape.

        Args:
            X: Features d'entraînement
            y: Target (optionnel)

        Returns:
            X_transformed: Données transformées
        """
        self.fit(X, y)
        return self.transform(X)

    def save(self, save_dir: str = './saved_models') -> str:
        """
        Sauvegarde le preprocessor dans un fichier.

        Args:
            save_dir: Répertoire de sauvegarde

        Returns:
            str: Chemin du fichier sauvegardé
        """
        if not self.is_fitted:
            raise RuntimeError("Le preprocessor doit être fitted avant d'être sauvegardé")

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"preprocessor_{timestamp}.joblib"
        filepath = os.path.join(save_dir, filename)

        joblib.dump(self, filepath)
        return filepath

    @staticmethod
    def load(filepath: str) -> 'DataPreprocessor':
        """
        Charge un preprocessor depuis un fichier.

        Args:
            filepath: Chemin vers le fichier

        Returns:
            DataPreprocessor: Instance chargée
        """
        return joblib.load(filepath)


def train_valid_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    train_size: float = 0.7,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    task_type: Optional[str] = None
) -> dict:
    """
    Split les données en train/valid/test.

    Args:
        X: Features
        y: Target
        train_size: Proportion pour l'entraînement (défaut: 0.7)
        valid_size: Proportion pour la validation (défaut: 0.15)
        test_size: Proportion pour le test (défaut: 0.15)
        random_state: Graine aléatoire pour reproductibilité
        task_type: Type de tâche ('classification' ou 'regression').
                  Si 'classification', stratification automatique

    Returns:
        dict: Dictionnaire contenant X_train, X_valid, X_test,
              y_train, y_valid, y_test

    Raises:
        ValueError: Si les proportions ne somment pas à 1.0
    """
    # Vérifier que les proportions somment à 1.0 (avec tolérance pour erreurs float)
    total = train_size + valid_size + test_size
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(
            f"Les proportions doivent sommer à 1.0. "
            f"Reçu: {train_size} + {valid_size} + {test_size} = {total}"
        )

    # Déterminer si on doit stratifier
    stratify_y = None
    if task_type == 'classification':
        stratify_y = y

    # Premier split: séparer train du reste (valid + test)
    temp_size = valid_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=temp_size,
        random_state=random_state,
        stratify=stratify_y
    )

    # Deuxième split: séparer valid et test
    # La proportion relative entre valid et test
    relative_test_size = test_size / temp_size

    stratify_temp = None
    if task_type == 'classification':
        stratify_temp = y_temp

    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=stratify_temp
    )

    return {
        'X_train': X_train,
        'X_valid': X_valid,
        'X_test': X_test,
        'y_train': y_train,
        'y_valid': y_valid,
        'y_test': y_test
    }
