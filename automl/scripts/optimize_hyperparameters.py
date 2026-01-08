"""
Script pour lancer l'optimisation sur Skinner.
Usage: python scripts/optimize_hyperparameters.py --data-path ...
"""
import argparse
import sys
import os
import joblib

# Hack pour importer automl si on lance depuis la racine
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl import fit, get_trained_models
from automl.core import get_data
from automl.optimization import HyperparameterTuner, OptimizationPipeline

def main():
    parser = argparse.ArgumentParser(description='Optimisation Hyperparamètres Skinner')
    parser.add_argument('--data-path', type=str, required=True, help='Chemin dataset')
    parser.add_argument('--search-method', type=str, default='random', choices=['grid', 'random'])
    parser.add_argument('--n-iter', type=int, default=20)
    parser.add_argument('--cv', type=int, default=5)
    parser.add_argument('--output', type=str, default='results/optimization_results.pkl')
    parser.add_argument('--reduced-space', action='store_true', help='Mode rapide')
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Lancement Optimisation : {args.search_method.upper()}")
    
    # 1. Pipeline complet initial
    print("--- Chargement & Entraînement Initial ---")
    fit(args.data_path, verbose=True)
    
    # 2. Récupération
    trainer = get_trained_models()
    models_dict = trainer.trained_models
    data = get_data()
    
    # 3. Tuning
    tuner = HyperparameterTuner(
        search_method=args.search_method,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=-1,  # Utilise tous les coeurs alloués par SLURM
        verbose=2
    )
    
    pipeline = OptimizationPipeline(tuner)
    optimized_models = pipeline.run(
        models_dict,
        data['X_train'], data['y_train'],
        data['X_valid'], data['y_valid'],
        use_reduced_space=args.reduced_space
    )
    
    # 4. Sauvegarde & Rapport
    tuner.save_results(args.output)
    print("\nTABLEAU FINAL :")
    print(pipeline.get_comparison(models_dict, optimized_models))

if __name__ == '__main__':
    main()