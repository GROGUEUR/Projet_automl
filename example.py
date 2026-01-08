"""
Script d'exemple d'utilisation du système AutoML.

Ce script démontre comment utiliser l'interface simple de AutoML
pour charger, prétraiter et préparer des données pour l'entraînement.
"""
import numpy as np
import pandas as pd
import os
import tempfile


def main():
    """
    Fonction principale de démonstration.
    """
    print("=" * 70)
    print("DÉMONSTRATION DU SYSTÈME AUTOML")
    print("=" * 70)
    print()

    # Créer un dataset d'exemple
    print("Création d'un dataset d'exemple...")
    data_path = create_sample_dataset()
    print(f"Dataset créé: {data_path}")
    print()

    # Importer AutoML
    print("Import du module AutoML...")
    import automl
    print(f"Version: {automl.__version__}")
    print()

    # Utilisation basique
    print("=" * 70)
    print("1. UTILISATION BASIQUE")
    print("=" * 70)
    print()

    print("Appel de automl.fit()...")
    automl.fit(data_path=data_path, verbose=True)
    print()

    # Accéder aux données
    print("=" * 70)
    print("2. ACCÈS AUX DONNÉES PRÉTRAITÉES")
    print("=" * 70)
    print()

    data = automl.get_data()
    print(f"Shape de X_train: {data['X_train'].shape}")
    print(f"Shape de X_valid: {data['X_valid'].shape}")
    print(f"Shape de X_test: {data['X_test'].shape}")
    print(f"Type de tâche détecté: {data['task_type']}")
    print()

    # Afficher quelques statistiques
    print("Statistiques de y_train:")
    print(f"  - Classe 0: {np.sum(data['y_train'] == 0)} échantillons")
    print(f"  - Classe 1: {np.sum(data['y_train'] == 1)} échantillons")
    print()

    # Évaluation (pour l'instant juste un placeholder)
    print("=" * 70)
    print("3. ÉVALUATION")
    print("=" * 70)
    print()
    print("Appel de automl.eval()...")
    results = automl.eval(verbose=True)
    print()

    # Utilisation avancée avec paramètres personnalisés
    print("=" * 70)
    print("4. UTILISATION AVANCÉE (PARAMÈTRES PERSONNALISÉS)")
    print("=" * 70)
    print()

    # Réinitialiser l'état
    automl.reset()

    # Nettoyer
    print("=" * 70)
    print("Nettoyage...")
    os.unlink(data_path)
    print(f"Fichier temporaire supprimé: {data_path}")
    print()

    print("=" * 70)
    print("DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
    print("=" * 70)
    print()
    print("Le système AutoML est prêt à être utilisé.")
    print("Les autres modules (models, optimization, evaluation) peuvent")
    print("maintenant être intégrés par les autres membres de l'équipe.")


if __name__ == '__main__':
    main()
