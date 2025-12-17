"""
Module de création (factory) des modèles sklearn.

Fournit des méthodes pour créer automatiquement des modèles
selon le type de tâche (classification ou régression).
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB

from .base_model import BaseModel


class ModelFactory:
    """
    Factory pour créer les modèles appropriés selon le type de tâche.
    """

    @staticmethod
    def get_default_models(task_type: str, random_state=42):
        """
        Retourne une liste de modèles par défaut pour le type de tâche.

        Args:
            task_type: 'classification' ou 'regression'
            random_state: seed pour reproductibilité

        Returns:
            List[BaseModel]: liste de modèles instanciés

        Raises:
            ValueError: si task_type n'est pas valide
        """
        models = []

        if task_type == 'classification':
            models = [
                BaseModel(
                    name="RandomForest",
                    model=RandomForestClassifier(random_state=random_state, n_jobs=-1),
                    task_type=task_type
                ),
                BaseModel(
                    name="GradientBoosting",
                    model=GradientBoostingClassifier(random_state=random_state),
                    task_type=task_type
                ),
                BaseModel(
                    name="LogisticRegression",
                    model=LogisticRegression(random_state=random_state, max_iter=1000),
                    task_type=task_type
                ),
                BaseModel(
                    name="SVM",
                    model=SVC(random_state=random_state, probability=True),
                    task_type=task_type
                ),
                BaseModel(
                    name="KNN",
                    model=KNeighborsClassifier(n_jobs=-1),
                    task_type=task_type
                ),
                BaseModel(
                    name="DecisionTree",
                    model=DecisionTreeClassifier(random_state=random_state),
                    task_type=task_type
                ),
                BaseModel(
                    name="NaiveBayes",
                    model=GaussianNB(),
                    task_type=task_type
                ),
            ]

        elif task_type == 'regression':
            models = [
                BaseModel(
                    name="RandomForest",
                    model=RandomForestRegressor(random_state=random_state, n_jobs=-1),
                    task_type=task_type
                ),
                BaseModel(
                    name="GradientBoosting",
                    model=GradientBoostingRegressor(random_state=random_state),
                    task_type=task_type
                ),
                BaseModel(
                    name="Ridge",
                    model=Ridge(random_state=random_state),
                    task_type=task_type
                ),
                BaseModel(
                    name="SVR",
                    model=SVR(),
                    task_type=task_type
                ),
                BaseModel(
                    name="KNN",
                    model=KNeighborsRegressor(n_jobs=-1),
                    task_type=task_type
                ),
                BaseModel(
                    name="DecisionTree",
                    model=DecisionTreeRegressor(random_state=random_state),
                    task_type=task_type
                ),
            ]

        else:
            raise ValueError(f"task_type invalide: {task_type}. Utilisez 'classification' ou 'regression'.")

        return models

    @staticmethod
    def create_model(model_name: str, task_type: str, random_state: int = 42, **kwargs):
        """
        Crée un modèle spécifique par son nom.

        Args:
            model_name: nom du modèle ('RandomForest', 'GradientBoosting', etc.)
            task_type: 'classification' ou 'regression'
            random_state: seed pour reproductibilité
            **kwargs: hyperparamètres du modèle sklearn

        Returns:
            BaseModel: modèle instancié

        Raises:
            ValueError: si le nom du modèle ou le task_type n'est pas valide
        """
        # Vérifier d'abord que le modèle existe
        available_models = ModelFactory.get_available_models(task_type)
        if model_name not in available_models:
            raise ValueError(
                f"Modèle '{model_name}' non reconnu pour {task_type}. "
                f"Modèles disponibles: {available_models}"
            )

        # Instancier uniquement le modèle demandé
        if task_type == 'classification':
            if model_name == 'RandomForest':
                sklearn_model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **kwargs)
            elif model_name == 'GradientBoosting':
                sklearn_model = GradientBoostingClassifier(random_state=random_state, **kwargs)
            elif model_name == 'LogisticRegression':
                sklearn_model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
            elif model_name == 'SVM':
                sklearn_model = SVC(random_state=random_state, probability=True, **kwargs)
            elif model_name == 'KNN':
                sklearn_model = KNeighborsClassifier(n_jobs=-1, **kwargs)
            elif model_name == 'DecisionTree':
                sklearn_model = DecisionTreeClassifier(random_state=random_state, **kwargs)
            elif model_name == 'NaiveBayes':
                sklearn_model = GaussianNB(**kwargs)

        elif task_type == 'regression':
            if model_name == 'RandomForest':
                sklearn_model = RandomForestRegressor(random_state=random_state, n_jobs=-1, **kwargs)
            elif model_name == 'GradientBoosting':
                sklearn_model = GradientBoostingRegressor(random_state=random_state, **kwargs)
            elif model_name == 'Ridge':
                sklearn_model = Ridge(random_state=random_state, **kwargs)
            elif model_name == 'SVR':
                sklearn_model = SVR(**kwargs)
            elif model_name == 'KNN':
                sklearn_model = KNeighborsRegressor(n_jobs=-1, **kwargs)
            elif model_name == 'DecisionTree':
                sklearn_model = DecisionTreeRegressor(random_state=random_state, **kwargs)

        return BaseModel(name=model_name, model=sklearn_model, task_type=task_type)

    @staticmethod
    def get_available_models(task_type: str):
        """
        Retourne la liste des noms de modèles disponibles pour un type de tâche.

        Args:
            task_type: 'classification' ou 'regression'

        Returns:
            List[str]: liste des noms de modèles disponibles
        """
        if task_type == 'classification':
            return ['RandomForest', 'GradientBoosting', 'LogisticRegression',
                    'SVM', 'KNN', 'DecisionTree', 'NaiveBayes']
        elif task_type == 'regression':
            return ['RandomForest', 'GradientBoosting', 'Ridge',
                    'SVR', 'KNN', 'DecisionTree']
        else:
            raise ValueError(f"task_type invalide: {task_type}")
