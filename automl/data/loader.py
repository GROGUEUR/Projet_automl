"""
Module de chargement des données pour AutoML.
"""
import os
import numpy as np
import pandas as pd
from typing import Tuple, Union


class DataLoader:
    """
    Classe pour charger les données depuis différents formats.

    Cette classe gère le chargement automatique de données depuis différents
    formats de fichiers (CSV, TXT) et détecte automatiquement le type de tâche
    (classification ou régression).
    """

    def __init__(self, data_path: str):
        """
        Initialise le DataLoader.

        Args:
            data_path: Chemin vers le répertoire ou fichier de données

        Raises:
            FileNotFoundError: Si le chemin spécifié n'existe pas
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le chemin {data_path} n'existe pas")

        self.data_path = data_path
        self.X = None
        self.y = None
        self.task_type = None

    def load_data(self) -> Tuple[Union[np.ndarray, pd.DataFrame],
                                  Union[np.ndarray, pd.Series], str]:
        """
        Charge les données et retourne X, y et le type de tâche.
        Détecte automatiquement le format (csv, txt, etc.).

        Returns:
            X: np.ndarray ou pd.DataFrame - features
            y: np.ndarray ou pd.Series - target (dernière colonne)
            task_type: str - 'classification' ou 'regression'

        Raises:
            ValueError: Si le format n'est pas supporté ou si les données sont invalides
        """
        # Déterminer si c'est un fichier ou un répertoire
        if os.path.isdir(self.data_path):
            # Vérifier si c'est un répertoire ChallengeMachineLearning (format .data + .solution)
            if self._is_challenge_format(self.data_path):
                self.X, self.y = self._load_challenge_format(self.data_path)

                # Vérifier les valeurs manquantes
                missing_count = np.sum(pd.isna(self.X)) + np.sum(pd.isna(self.y))
                if missing_count > 0:
                    print(f"⚠️  Attention: {missing_count} valeurs manquantes détectées")

                # Détecter le type de tâche
                self.task_type = self.detect_task_type(self.y)

                return self.X, self.y, self.task_type
            else:
                # Chercher un fichier de données dans le répertoire (ancien format)
                data_file = self._find_data_file(self.data_path)
        else:
            data_file = self.data_path

        # Charger selon l'extension
        file_extension = os.path.splitext(data_file)[1].lower()

        if file_extension == '.csv':
            data = self._load_csv(data_file)
        elif file_extension in ['.txt', '.dat', '.data']:
            data = self._load_txt(data_file)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_extension}")

        # Vérifier qu'on a bien des données
        if data is None or len(data) == 0:
            raise ValueError("Aucune donnée chargée")

        # Séparer features et target (dernière colonne = target)
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values

        # Vérifier les valeurs manquantes
        missing_count = np.sum(pd.isna(self.X)) + np.sum(pd.isna(self.y))
        if missing_count > 0:
            print(f"⚠️  Attention: {missing_count} valeurs manquantes détectées")

        # Détecter le type de tâche
        self.task_type = self.detect_task_type(self.y)

        return self.X, self.y, self.task_type

    def _is_challenge_format(self, directory: str) -> bool:
        """
        Vérifie si le répertoire contient des fichiers au format ChallengeMachineLearning.
        Format attendu: data_X.data + data_X.solution (+ optionnel data_X.type)

        Args:
            directory: Chemin vers le répertoire

        Returns:
            bool: True si le format Challenge est détecté
        """
        files = os.listdir(directory)
        # Chercher un fichier .data et un fichier .solution
        has_data = any(f.endswith('.data') for f in files)
        has_solution = any(f.endswith('.solution') for f in files)
        return has_data and has_solution

    def _load_challenge_format(self, directory: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Charge les données au format ChallengeMachineLearning.
        Structure: data_X.data (features) + data_X.solution (target) + data_X.type (optionnel)

        Args:
            directory: Chemin vers le répertoire contenant les fichiers

        Returns:
            Tuple[np.ndarray, np.ndarray]: X (features), y (target)

        Raises:
            FileNotFoundError: Si les fichiers .data ou .solution ne sont pas trouvés
        """
        files = os.listdir(directory)

        # Trouver le fichier .data
        data_file = None
        solution_file = None

        for f in files:
            if f.endswith('.data'):
                data_file = os.path.join(directory, f)
            if f.endswith('.solution'):
                solution_file = os.path.join(directory, f)

        if data_file is None:
            raise FileNotFoundError(f"Fichier .data non trouvé dans {directory}")
        if solution_file is None:
            raise FileNotFoundError(f"Fichier .solution non trouvé dans {directory}")

        # Charger X (features) - séparé par des espaces
        X = pd.read_csv(data_file, sep=' ', header=None).values

        # Charger y (target)
        y = pd.read_csv(solution_file, header=None, names=['target'])['target'].values

        return X, y

    def _find_data_file(self, directory: str) -> str:
        """
        Trouve le premier fichier de données dans un répertoire.

        Args:
            directory: Chemin vers le répertoire

        Returns:
            str: Chemin vers le fichier de données trouvé

        Raises:
            FileNotFoundError: Si aucun fichier de données n'est trouvé
        """
        supported_extensions = ['.csv', '.txt', '.dat']

        for file in os.listdir(directory):
            if any(file.endswith(ext) for ext in supported_extensions):
                return os.path.join(directory, file)

        raise FileNotFoundError(
            f"Aucun fichier de données trouvé dans {directory}. "
            f"Extensions supportées: {supported_extensions}"
        )

    def _load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Charge un fichier CSV.

        Args:
            file_path: Chemin vers le fichier CSV

        Returns:
            pd.DataFrame: Données chargées
        """
        # Essayer différents séparateurs
        separators = [',', ';', '\t', ' ']

        for sep in separators:
            try:
                data = pd.read_csv(file_path, sep=sep, header=None)
                # Vérifier qu'on a au moins 2 colonnes (features + target)
                if data.shape[1] >= 2:
                    return data
            except Exception:
                continue

        # Si aucun séparateur ne fonctionne, essayer sans spécifier
        try:
            data = pd.read_csv(file_path, header=None)
            if data.shape[1] >= 2:
                return data
        except Exception as e:
            raise ValueError(f"Impossible de charger le fichier CSV: {e}")

        raise ValueError("Format CSV non reconnu")

    def _load_txt(self, file_path: str) -> pd.DataFrame:
        """
        Charge un fichier TXT avec différents séparateurs possibles.

        Args:
            file_path: Chemin vers le fichier TXT

        Returns:
            pd.DataFrame: Données chargées
        """
        # Essayer différents séparateurs
        separators = ['\t', ' ', ',', ';', '\\s+']

        for sep in separators:
            try:
                if sep == '\\s+':
                    # Regex pour espaces multiples
                    data = pd.read_csv(file_path, sep=sep, header=None,
                                     engine='python', skipinitialspace=True)
                else:
                    data = pd.read_csv(file_path, sep=sep, header=None)

                # Vérifier qu'on a au moins 2 colonnes
                if data.shape[1] >= 2:
                    return data
            except Exception:
                continue

        raise ValueError("Impossible de détecter le séparateur du fichier TXT")

    def detect_task_type(self, y: Union[np.ndarray, pd.Series]) -> str:
        """
        Détecte si c'est une tâche de classification ou régression.

        Critères de détection:
        - Classification: type object/string, ou nombre de valeurs uniques < 20
          et représentant < 5% du total
        - Régression: type numérique avec beaucoup de valeurs différentes

        Args:
            y: Vecteur target

        Returns:
            str: 'classification' ou 'regression'
        """
        # Convertir en pandas Series pour faciliter l'analyse
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y

        # Si le type est object/string, c'est de la classification
        if y_series.dtype == 'object' or y_series.dtype.name == 'category':
            return 'classification'

        # Compter le nombre de valeurs uniques
        n_unique = y_series.nunique()
        n_total = len(y_series)

        # Règle heuristique:
        # - Si moins de 20 valeurs uniques ET moins de 5% du total -> classification
        # - Sinon -> régression
        if n_unique < 20 and (n_unique / n_total) < 0.05:
            return 'classification'
        else:
            return 'regression'

    def get_info(self) -> dict:
        """
        Retourne les informations sur les données chargées.

        Returns:
            dict: Dictionnaire contenant les statistiques des données
        """
        if self.X is None:
            return {"status": "Aucune donnée chargée"}

        return {
            "n_samples": self.X.shape[0],
            "n_features": self.X.shape[1],
            "task_type": self.task_type,
            "n_classes": len(np.unique(self.y)) if self.task_type == 'classification' else None,
            "missing_values": np.sum(pd.isna(self.X)) + np.sum(pd.isna(self.y))
        }
