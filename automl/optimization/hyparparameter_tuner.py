from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, r2_score
import numpy as np
import time
from typing import Dict, Any
import joblib

from .hyperparameter_space import HyperparameterSpace

class HyperparameterTuner:
    """
    Optimise les hyperparamètres des modèles.
    """
    
    def __init__(self, search_method='grid', n_iter=20, cv=5, 
                 n_jobs=-1, verbose=1, random_state=42):
        self.search_method = search_method
        self.n_iter = n_iter
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        
        self.best_params = {}
        self.search_results = {}
    
    def optimize(self, model, X_train, y_train, param_space=None):
        """
        Optimise les hyperparamètres d'un modèle unique.
        """
        if param_space is None:
            param_space = HyperparameterSpace.get_space(
                model.name, 
                model.task_type
            )
        
        if not param_space:
            if self.verbose:
                print(f"⚠️  Pas d'espace de recherche défini pour {model.name}")
            return model.get_params()
        
        # Définir le scorer selon le type de tâche
        if model.task_type == 'classification':
            scorer = make_scorer(accuracy_score)
        else:
            scorer = make_scorer(r2_score)
        
        # Choisir la méthode de recherche
        if self.search_method == 'grid':
            search = GridSearchCV(
                estimator=model.model,
                param_grid=param_space,
                cv=self.cv,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                return_train_score=True
            )
        
        elif self.search_method == 'random':
            search = RandomizedSearchCV(
                estimator=model.model,
                param_distributions=param_space,
                n_iter=self.n_iter,
                cv=self.cv,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            raise ValueError(f"Méthode de recherche inconnue: {self.search_method}")
        
        if self.verbose:
            print(f"\n🔍 Optimisation de {model.name} avec {self.search_method}Search...")
        
        # Lancer la recherche
        start_time = time.time()
        search.fit(X_train, y_train)
        search_time = time.time() - start_time
        
        # Stocker les résultats
        self.best_params[model.name] = search.best_params_
        self.search_results[model.name] = {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'search_time': search_time,
            'cv_results': search.cv_results_
        }
        
        if self.verbose:
            print(f"✓ Terminé en {search_time:.2f}s")
            print(f"✓ Meilleur score CV: {search.best_score_:.4f}")
            print(f"✓ Meilleurs paramètres: {search.best_params_}")
        
        return search.best_params_
    
    def optimize_all(self, models_dict, X_train, y_train, use_reduced_space=False):
        """
        Optimise tous les modèles d'un dictionnaire.
        """
        optimized_params = {}
        
        for name, model in models_dict.items():
            try:
                if use_reduced_space:
                    param_space = HyperparameterSpace.get_reduced_space(
                        model.name, 
                        model.task_type
                    )
                else:
                    param_space = None # Utilise l'espace complet
                
                best_params = self.optimize(model, X_train, y_train, param_space)
                optimized_params[name] = best_params
                
            except Exception as e:
                if self.verbose:
                    print(f"✗ Erreur lors de l'optimisation de {name}: {e}")
                continue
        
        return optimized_params
    
    def save_results(self, filepath: str):
        joblib.dump(self.search_results, filepath)
        if self.verbose:
            print(f"✓ Résultats sauvegardés: {filepath}")
    
    def load_results(self, filepath: str):
        self.search_results = joblib.load(filepath)
        self.best_params = {name: res['best_params'] 
                           for name, res in self.search_results.items()}