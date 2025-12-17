"""
Visualisation des résultats d'évaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

class ResultsVisualizer:
    """
    Crée des visualisations pour les résultats d'évaluation.
    """
    
    def __init__(self, style='seaborn-v0_8', figsize=(10, 6)):
        plt.style.use(style)
        self.figsize = figsize
        sns.set_palette("husl")
    
    def plot_model_comparison(self, comparison_df, task_type, 
                             save_path=None):
        """
        Graphique comparatif des performances des modèles.
        
        Args:
            comparison_df: DataFrame de comparaison
            task_type: 'classification' ou 'regression'
            save_path: chemin de sauvegarde (optionnel)
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if task_type == 'classification':
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        else:
            metrics = ['R²', 'RMSE', 'MAE']
        
        # Préparer les données
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        # Créer le graphique en barres groupées
        x = np.arange(len(comparison_df))
        width = 0.8 / len(available_metrics)
        
        for i, metric in enumerate(available_metrics):
            offset = width * i - width * (len(available_metrics) - 1) / 2
            ax.bar(x + offset, comparison_df[metric], width, label=metric)
        
        ax.set_xlabel('Modèle', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des performances', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, 
                             save_path=None):
        """
        Matrice de confusion.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, ax=ax, cmap='Blues',
            colorbar=True
        )
        
        ax.set_title(f'Matrice de confusion - {model_name}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Matrice de confusion sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba, model_name, 
                      save_path=None):
        """
        Courbe ROC (classification binaire uniquement).
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        RocCurveDisplay.from_predictions(
            y_true, y_pred_proba[:, 1], ax=ax, name=model_name
        )
        
        ax.plot([0, 1], [0, 1], 'k--', label='Aléatoire')
        ax.set_title(f'Courbe ROC - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Courbe ROC sauvegardée: {save_path}")
        
        plt.show()
    
    def plot_residuals(self, y_true, y_pred, model_name, save_path=None):
        """
        Graphique des résidus (régression).
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Résidus vs prédictions
        axes[0].scatter(y_pred, residuals, alpha=0.6)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Valeurs prédites', fontweight='bold')
        axes[0].set_ylabel('Résidus', fontweight='bold')
        axes[0].set_title('Résidus vs Prédictions', fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Distribution des résidus
        axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Résidus', fontweight='bold')
        axes[1].set_ylabel('Fréquence', fontweight='bold')
        axes[1].set_title('Distribution des résidus', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        fig.suptitle(f'{model_name} - Analyse des résidus', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique des résidus sauvegardé: {save_path}")
        
        plt.show()
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, 
                                   save_path=None):
        """
        Prédictions vs valeurs réelles (régression).
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.scatter(y_true, y_pred, alpha=0.6)
        
        # Ligne de référence parfaite
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, label='Prédiction parfaite')
        
        ax.set_xlabel('Valeurs réelles', fontweight='bold')
        ax.set_ylabel('Valeurs prédites', fontweight='bold')
        ax.set_title(f'{model_name} - Prédictions vs Réalité', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Graphique sauvegardé: {save_path}")
        
        plt.show()
