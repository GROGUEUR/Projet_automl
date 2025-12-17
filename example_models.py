"""
Script d'exemple pour tester le module models indépendamment.

Ce script démontre l'utilisation complète du module models (Personne 2)
avec des données synthétiques.
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Import du module models
from automl.models import ModelFactory, ModelTrainer, ModelSelector


def test_classification():
    """Test complet sur un problème de classification."""
    print("=" * 70)
    print("TEST DE CLASSIFICATION")
    print("=" * 70)

    # Générer des données synthétiques
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )

    # Séparer train/valid
    X_train, X_valid = X[:400], X[400:]
    y_train, y_valid = y[:400], y[400:]

    print(f"\nDonnées générées:")
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Valid: {X_valid.shape[0]} échantillons")
    print(f"  - Features: {X_train.shape[1]}")

    # Créer le trainer avec tous les modèles
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)

    trainer = ModelTrainer(task_type='classification', random_state=42, verbose=True)

    # Entraîner tous les modèles
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Afficher le résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 70)
    summary = trainer.get_results_summary()
    print(summary)

    # Sélectionner le meilleur
    best_model = trainer.select_best_model(metric='valid_score')

    # Tester différentes stratégies de sélection
    print("\n" + "=" * 70)
    print("STRATÉGIES DE SÉLECTION")
    print("=" * 70)

    # Par score
    best_by_score = ModelSelector.select_by_score(results, metric='valid_score')
    print(f"Meilleur par score: {best_by_score}")

    # Par compromis vitesse/score
    best_tradeoff = ModelSelector.select_by_speed_score_tradeoff(
        results, score_weight=0.7, speed_weight=0.3
    )
    print(f"Meilleur par compromis: {best_tradeoff}")

    # Top 3
    top_3 = ModelSelector.select_top_k(results, k=3)
    print(f"Top 3 modèles: {top_3}")

    # Avec contrôle du surapprentissage
    best_controlled = ModelSelector.select_by_overfitting_control(results, max_gap=0.15)
    print(f"Meilleur avec contrôle overfitting: {best_controlled}")

    return trainer


def test_regression():
    """Test complet sur un problème de régression."""
    print("\n\n" + "=" * 70)
    print("TEST DE RÉGRESSION")
    print("=" * 70)

    # Générer des données synthétiques
    X, y = make_regression(
        n_samples=500,
        n_features=20,
        random_state=42
    )

    # Séparer train/valid
    X_train, X_valid = X[:400], X[400:]
    y_train, y_valid = y[:400], y[400:]

    print(f"\nDonnées générées:")
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Valid: {X_valid.shape[0]} échantillons")
    print(f"  - Features: {X_train.shape[1]}")

    # Créer le trainer avec tous les modèles
    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)

    trainer = ModelTrainer(task_type='regression', random_state=42, verbose=True)

    # Entraîner tous les modèles
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Afficher le résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ DES PERFORMANCES")
    print("=" * 70)
    summary = trainer.get_results_summary()
    print(summary)

    # Sélectionner le meilleur
    best_model = trainer.select_best_model(metric='valid_score')

    return trainer


def test_model_factory():
    """Test de la factory de modèles."""
    print("\n\n" + "=" * 70)
    print("TEST DE MODEL FACTORY")
    print("=" * 70)

    # Afficher les modèles disponibles
    clf_models = ModelFactory.get_available_models('classification')
    print(f"\nModèles de classification disponibles: {clf_models}")

    reg_models = ModelFactory.get_available_models('regression')
    print(f"Modèles de régression disponibles: {reg_models}")

    # Créer un modèle personnalisé
    print("\nCréation d'un RandomForest personnalisé:")
    custom_rf = ModelFactory.create_model(
        'RandomForest',
        'classification',
        n_estimators=200,
        max_depth=10
    )
    print(f"  Modèle: {custom_rf.name}")
    print(f"  Paramètres: n_estimators={custom_rf.get_params()['n_estimators']}, "
          f"max_depth={custom_rf.get_params()['max_depth']}")


if __name__ == '__main__':
    # Test de la factory
    test_model_factory()

    # Test classification
    clf_trainer = test_classification()

    # Test régression
    reg_trainer = test_regression()

    print("\n\n" + "=" * 70)
    print("TOUS LES TESTS SONT RÉUSSIS !")
    print("=" * 70)
    print("\nLe module models (Personne 2) est prêt à être intégré.")
    print("Il peut maintenant être utilisé par:")
    print("  - Personne 3: pour l'optimisation des hyperparamètres")
    print("  - Personne 4: pour l'évaluation finale")
