import time
import pandas as pd
from ..models.base_model import BaseModel

class OptimizationPipeline:
    """
    Pipeline complet d'optimisation : optimiser puis ré-entraîner.
    """
    
    def __init__(self, tuner, verbose=True):
        self.tuner = tuner
        self.verbose = verbose
        self.optimized_models = {}
    
    def run(self, models_dict, X_train, y_train, X_valid, y_valid, 
            use_reduced_space=False):
        if self.verbose:
            print("\n" + "="*70)
            print("🚀 DÉBUT DE L'OPTIMISATION DES HYPERPARAMÈTRES")
            print("="*70)
        
        # 1. Optimiser les hyperparamètres
        best_params = self.tuner.optimize_all(
            models_dict, 
            X_train, 
            y_train,
            use_reduced_space
        )
        
        # 2. Ré-entraîner avec les meilleurs paramètres
        if self.verbose:
            print("\n" + "="*70)
            print("🔄 RÉ-ENTRAÎNEMENT AVEC HYPERPARAMÈTRES OPTIMISÉS")
            print("="*70)
        
        for name, model in models_dict.items():
            if name not in best_params:
                continue
            
            try:
                if self.verbose:
                    print(f"\n📊 Ré-entraînement de {name}...")
                
                # Créer une copie du modèle avec les nouveaux params
                # On suppose que model.model est l'estimateur sklearn
                new_estimator = model.model.__class__(**best_params[name])
                
                optimized_model = BaseModel(
                    name=f"{name}_optimized",
                    model=new_estimator,
                    task_type=model.task_type
                )
                
                # Entraîner sur train + valid idéalement, mais ici on garde train
                # pour comparer équitablement sur valid
                start_time = time.time()
                optimized_model.fit(X_train, y_train, X_valid, y_valid)
                
                self.optimized_models[name] = optimized_model
                
                if self.verbose:
                    improvement = 0
                    if model.valid_score is not None:
                        improvement = optimized_model.valid_score - model.valid_score
                    print(f"✓ Score validation: {optimized_model.valid_score:.4f} (Amélioration: {improvement:+.4f})")
            
            except Exception as e:
                if self.verbose:
                    print(f"✗ Erreur: {e}")
                continue
        
        return self.optimized_models

    def get_comparison(self, original_models, optimized_models):
        """Retourne un DataFrame comparatif."""
        comparison = []
        for name in original_models.keys():
            if name in optimized_models:
                orig = original_models[name]
                optim = optimized_models[name]
                
                comparison.append({
                    'Model': name,
                    'Original_Score': orig.valid_score,
                    'Optimized_Score': optim.valid_score,
                    'Improvement': optim.valid_score - orig.valid_score
                })
        
        if not comparison:
            return pd.DataFrame()
        return pd.DataFrame(comparison).sort_values('Improvement', ascending=False)