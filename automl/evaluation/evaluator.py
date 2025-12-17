from .metrics import MetricsCalculator
from ..models.base_model import BaseModel
from typing import Dict, List
import pandas as pd
import numpy as np

class ModelEvaluator:
    """
    Évalue les performances des modèles sur différents ensembles de données.
    """
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results = {}
    
    def evaluate_model(self, model: BaseModel, X, y, dataset_name='test'):
        """
        Évalue un modèle sur un ensemble de données.
        
        Args:
            model: modèle à évaluer
            X, y: données de test
            dataset_name: nom de l'ensemble ('train', 'valid', 'test')
        
        Returns:
            Dict: métriques d'évaluation
        """
        if not model.is_fitted:
            raise ValueError(f"Le modèle {model.name} n'est pas entraîné.")
        
        # Prédictions
        y_pred = model.predict(X)
        
        # Probabilités (si classification)
        y_pred_proba = None
        if model.task_type == 'classification':
            try:
                y_pred_proba = model.predict_proba(X)
            except:
                pass
        
        # Calculer les métriques
        metrics = MetricsCalculator.compute_metrics(
            y, y_pred, model.task_type, y_pred_proba
        )
        
        # Stocker les résultats
        result = {
            'model_name': model.name,
            'task_type': model.task_type,
            'dataset': dataset_name,
            'metrics': metrics,
            'y_true': y,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        key = f"{model.name}_{dataset_name}"
        self.results[key] = result
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 Évaluation de {model.name} sur {dataset_name}")
            print(f"{'='*60}")
            self._print_metrics(metrics, model.task_type)
        
        return result
    
    def evaluate_all(self, models_dict: Dict[str, BaseModel], 
                    X_test, y_test, X_valid=None, y_valid=None):
        """
        Évalue tous les modèles sur les ensembles de test (et validation).
        
        Args:
            models_dict: Dict[str, BaseModel]
            X_test, y_test: données de test
            X_valid, y_valid: données de validation (optionnel)
        
        Returns:
            Dict: résultats pour tous les modèles
        """
        all_results = {}
        
        for name, model in models_dict.items():
            try:
                # Évaluation sur test
                test_result = self.evaluate_model(model, X_test, y_test, 'test')
                all_results[f"{name}_test"] = test_result
                
                # Évaluation sur validation si fourni
                if X_valid is not None and y_valid is not None:
                    valid_result = self.evaluate_model(model, X_valid, y_valid, 'valid')
                    all_results[f"{name}_valid"] = valid_result
                
            except Exception as e:
                if self.verbose:
                    print(f"✗ Erreur lors de l'évaluation de {name}: {e}")
                continue
        
        self.results.update(all_results)
        return all_results
    
    def get_comparison_table(self, dataset='test'):
        """
        Crée un tableau comparatif des performances.
        
        Args:
            dataset: 'test' ou 'valid'
        
        Returns:
            pd.DataFrame: tableau comparatif
        """
        comparison = []
        
        for key, result in self.results.items():
            if dataset in key:
                metrics = result['metrics']
                task_type = result['task_type']
                
                if task_type == 'classification':
                    row = {
                        'Model': result['model_name'],
                        'Accuracy': metrics.get('accuracy', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1-Score': metrics.get('f1_score', 0),
                        'ROC-AUC': metrics.get('roc_auc', 0) or 0
                    }
                else:  # regression
                    row = {
                        'Model': result['model_name'],
                        'R²': metrics.get('r2_score', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0),
                        'MAPE': metrics.get('mape', 0) or 0
                    }
                
                comparison.append(row)
        
        df = pd.DataFrame(comparison)
        
        # Trier par métrique principale
        if not df.empty:
            if 'Accuracy' in df.columns:
                df = df.sort_values('Accuracy', ascending=False)
            elif 'R²' in df.columns:
                df = df.sort_values('R²', ascending=False)
        
        return df
    
    def _print_metrics(self, metrics, task_type):
        """Affiche les métriques de manière formatée."""
        table = MetricsCalculator.format_metrics_table(metrics, task_type)
        print(table.to_string())
    
    def save_results(self, filepath):
        """Sauvegarde les résultats d'évaluation."""
        import joblib
        joblib.dump(self.results, filepath)
        if self.verbose:
            print(f"✓ Résultats sauvegardés: {filepath}")
