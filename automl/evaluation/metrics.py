"""
Métriques d'évaluation pour classification et régression.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any
import pandas as pd

class MetricsCalculator:
    """
    Calcule toutes les métriques pertinentes selon le type de tâche.
    """
    
    @staticmethod
    def compute_classification_metrics(y_true, y_pred, y_pred_proba=None, 
                                      average='weighted'):
        """
        Calcule toutes les métriques de classification.
        
        Args:
            y_true: vraies étiquettes
            y_pred: prédictions
            y_pred_proba: probabilités (optionnel, pour ROC-AUC)
            average: 'binary', 'weighted', 'macro', 'micro'
        
        Returns:
            Dict: toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, 
                                               average=average, 
                                               zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, 
                                        average=average, 
                                        zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, 
                                      average=average, 
                                      zero_division=0)
        
        # ROC-AUC (si probabilités disponibles)
        if y_pred_proba is not None:
            try:
                # Pour binaire ou multiclasse
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                       multi_class='ovr', 
                                                       average=average)
            except:
                metrics['roc_auc'] = None
        
        # Matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Rapport de classification détaillé
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            output_dict=True,
            zero_division=0
        )
        
        return metrics
    
    @staticmethod
    def compute_regression_metrics(y_true, y_pred):
        """
        Calcule toutes les métriques de régression.
        
        Args:
            y_true: vraies valeurs
            y_pred: prédictions
        
        Returns:
            Dict: toutes les métriques
        """
        metrics = {}
        
        # Métriques de base
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2_score'] = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Si division par zéro
            mask = y_true != 0
            if mask.any():
                metrics['mape'] = np.mean(np.abs((y_true[mask] - y_pred[mask]) 
                                                 / y_true[mask])) * 100
            else:
                metrics['mape'] = None
        
        # Erreur maximale
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # Coefficient de variation de l'erreur
        metrics['cv_rmse'] = (metrics['rmse'] / np.mean(y_true)) * 100 if np.mean(y_true) != 0 else None
        
        return metrics
    
    @staticmethod
    def compute_metrics(y_true, y_pred, task_type, y_pred_proba=None):
        """
        Calcule les métriques appropriées selon le type de tâche.
        
        Args:
            y_true: vraies valeurs
            y_pred: prédictions
            task_type: 'classification' ou 'regression'
            y_pred_proba: probabilités (classification uniquement)
        
        Returns:
            Dict: métriques calculées
        """
        if task_type == 'classification':
            return MetricsCalculator.compute_classification_metrics(
                y_true, y_pred, y_pred_proba
            )
        elif task_type == 'regression':
            return MetricsCalculator.compute_regression_metrics(y_true, y_pred)
        else:
            raise ValueError(f"Type de tâche inconnu: {task_type}")
    
    @staticmethod
    def format_metrics_table(metrics_dict, task_type):
        """
        Formate les métriques en DataFrame pour affichage.
        
        Returns:
            pd.DataFrame: métriques formatées
        """
        if task_type == 'classification':
            display_metrics = {
                'Accuracy': metrics_dict.get('accuracy', 0),
                'Precision': metrics_dict.get('precision', 0),
                'Recall': metrics_dict.get('recall', 0),
                'F1-Score': metrics_dict.get('f1_score', 0),
                'ROC-AUC': metrics_dict.get('roc_auc', 0)
            }
        else:  # regression
            display_metrics = {
                'R² Score': metrics_dict.get('r2_score', 0),
                'RMSE': metrics_dict.get('rmse', 0),
                'MAE': metrics_dict.get('mae', 0),
                'MAPE (%)': metrics_dict.get('mape', 0),
                'Max Error': metrics_dict.get('max_error', 0)
            }
        
        return pd.DataFrame([display_metrics]).T.rename(columns={0: 'Value'})
