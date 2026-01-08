"""
Script de test pour vérifier le chargement au format ChallengeMachineLearning.

Ce script teste que le DataLoader peut charger des données depuis un répertoire
contenant data_X.data et data_X.solution.
"""
import automl

# ==========================================
# TEST 1: Format ChallengeMachineLearning
# ==========================================
print("="*70)
print("TEST: Format ChallengeMachineLearning")
print("="*70)
print()

# Exemple avec un dataset du Challenge
# Remplacez 'A' par la lettre de votre dataset (A, B, C, D, etc.)
DATASET = 'D'
BASE_PATH = "/info/corpus/ChallengeMachineLearning"

# Chemin complet vers le répertoire du dataset
data_path = f"{BASE_PATH}/data_{DATASET}"

print(f"📂 Chargement du dataset {DATASET}...")
print(f"   Chemin: {data_path}")
print()

try:
    # Utiliser AutoML avec le nouveau format
    automl.fit(
        data_path=data_path,
        verbose=True
    )

    print()
    print("="*70)
    print("✅ SUCCÈS: Le chargement au format Challenge fonctionne!")
    print("="*70)
    print()

    # Afficher les informations
    data = automl.get_data()
    print("📊 INFORMATIONS SUR LES DONNÉES:")
    print(f"   • Shape de X_train: {data['X_train'].shape}")
    print(f"   • Shape de y_train: {data['y_train'].shape}")
    print(f"   • Type de tâche: {data['task_type']}")
    print(f"   • Nombre de modèles entraînés: {len(data['trained_models'])}")
    print()

    # Évaluer
    print("="*70)
    print("ÉVALUATION DES MODÈLES")
    print("="*70)
    print()

    results = automl.eval(verbose=True)

except FileNotFoundError as e:
    print(f"❌ ERREUR: {e}")
    print()
    print("💡 SOLUTIONS:")
    print(f"   1. Vérifiez que le répertoire existe: {data_path}")
    print(f"   2. Vérifiez qu'il contient les fichiers:")
    print(f"      - data_{DATASET}.data")
    print(f"      - data_{DATASET}.solution")
    print(f"   3. Changez la variable DATASET en haut du script")
    print(f"   4. Changez la variable BASE_PATH si nécessaire")

except Exception as e:
    print(f"❌ ERREUR inattendue: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("FIN DU TEST")
print("="*70)
