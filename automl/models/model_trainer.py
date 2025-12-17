"""
Module d'entraînement et de gestion des modèles.

Fournit la classe ModelTrainer qui orchestre l'entraînement
de plusieurs modèles et la sélection du meilleur.
"""
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .base_model import BaseModel
from .model_factory import ModelFactory


class ModelTrainer:
    """
    Classe pour entraîner et comparer plusieurs modèles.
    """

    def __init__(self, task_type: str, models: Optional[List[BaseModel]] = None,
                 random_state: int = 42, verbose: bool = True):
        """
        Initialise le trainer.

        Args:
            task_type: 'classification' ou 'regression'
            models: liste de modèles (si None, utilise les modèles par défaut)
            random_state: seed pour reproductibilité
            verbose: afficher les logs
        """
        self.task_type = task_type
        self.random_state = random_state
        self.verbose = verbose

        if models is None:
            self.models = ModelFactory.get_default_models(task_type, random_state)
        else:
            self.models = models

        self.trained_models = {}
        self.best_model = None
        self.results = []

    def train_all(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Entraîne tous les modèles et conserve les résultats.

        Args:
            X_train, y_train: données d'entraînement
            X_valid, y_valid: données de validation

        Returns:
            List[Dict]: résultats de tous les modèles
        """
        self.results = []

        for model in self.models:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Entraînement de {model.name}...")
                print(f"{'='*60}")

            try:
                start_time = time.time()
                model.fit(X_train, y_train, X_valid, y_valid)
                training_time = time.time() - start_time

                self.trained_models[model.name] = model

                result = {
                    'name': model.name,
                    'train_score': model.train_score,
                    'valid_score': model.valid_score,
                    'training_time': training_time
                }
                self.results.append(result)

                if self.verbose:
                    print(f"✓ Score train: {model.train_score:.4f}")
                    if model.valid_score is not None:
                        print(f"✓ Score valid: {model.valid_score:.4f}")
                    print(f"✓ Temps: {training_time:.2f}s")

            except Exception as e:
                if self.verbose:
                    print(f"✗ Erreur lors de l'entraînement de {model.name}: {e}")
                continue

        return self.results

    def select_best_model(self, metric='valid_score'):
        """
        Sélectionne le meilleur modèle selon une métrique.

        Args:
            metric: métrique à utiliser ('valid_score', 'train_score')

        Returns:
            BaseModel: meilleur modèle

        Raises:
            ValueError: si aucun modèle n'est entraîné
        """
        if not self.results:
            raise ValueError("Aucun modèle entraîné. Appelez train_all() d'abord.")

        # Filtrer les résultats avec une métrique valide
        valid_results = [r for r in self.results if r.get(metric) is not None]

        if not valid_results:
            raise ValueError(f"Aucun modèle n'a de score valide pour la métrique '{metric}'")

        # Trier les résultats par score décroissant
        sorted_results = sorted(valid_results, key=lambda x: x[metric], reverse=True)
        best_result = sorted_results[0]

        self.best_model = self.trained_models[best_result['name']]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🏆 Meilleur modèle: {self.best_model.name}")
            print(f"   Score validation: {best_result['valid_score']:.4f}")
            print(f"{'='*60}\n")

        return self.best_model

    def get_results_summary(self):
        """
        Retourne un résumé des résultats sous forme de DataFrame.

        Returns:
            pd.DataFrame: résumé trié par score de validation
        """
        if not self.results:
            return pd.DataFrame()

        df = pd.DataFrame(self.results)

        # Trier par valid_score si disponible, sinon par train_score
        if 'valid_score' in df.columns and df['valid_score'].notna().any():
            df = df.sort_values('valid_score', ascending=False)
        elif 'train_score' in df.columns:
            df = df.sort_values('train_score', ascending=False)

        return df

    def save_all_models(self, save_dir: str):
        """
        Sauvegarde tous les modèles entraînés.

        Args:
            save_dir: répertoire de sauvegarde

        Returns:
            List[str]: chemins des fichiers sauvegardés
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        saved_paths = []
        for name, model in self.trained_models.items():
            try:
                path = model.save(save_dir)
                saved_paths.append(path)
            except Exception as e:
                if self.verbose:
                    print(f"⚠ Erreur lors de la sauvegarde de {name}: {e}")

        if self.verbose:
            print(f"✓ {len(saved_paths)} modèles sauvegardés dans {save_dir}")

        return saved_paths

    def save_best_model(self, save_dir: str):
        """
        Sauvegarde uniquement le meilleur modèle.

        Args:
            save_dir: répertoire de sauvegarde

        Returns:
            str: chemin du fichier sauvegardé

        Raises:
            ValueError: si aucun meilleur modèle n'a été sélectionné
        """
        if self.best_model is None:
            raise ValueError("Aucun meilleur modèle sélectionné. Appelez select_best_model() d'abord.")

        path = self.best_model.save(save_dir)

        if self.verbose:
            print(f"✓ Meilleur modèle sauvegardé: {path}")

        return path

    def get_model(self, name: str):
        """
        Récupère un modèle spécifique par son nom.

        Args:
            name: nom du modèle

        Returns:
            BaseModel: modèle demandé

        Raises:
            KeyError: si le modèle n'existe pas
        """
        if name not in self.trained_models:
            raise KeyError(f"Modèle '{name}' non trouvé. Modèles disponibles: {list(self.trained_models.keys())}")

        return self.trained_models[name]

    def get_all_models(self):
        """
        Retourne tous les modèles entraînés.

        Returns:
            Dict[str, BaseModel]: dictionnaire nom -> modèle
        """
        return self.trained_models

    def __repr__(self):
        """Représentation string du trainer."""
        n_models = len(self.trained_models)
        best_name = self.best_model.name if self.best_model else "None"
        return f"ModelTrainer(task_type='{self.task_type}', n_models={n_models}, best_model='{best_name}')"
