"""
Test d'intégration du module models avec l'interface automl.

Ce script teste que le module models s'intègre correctement
dans le pipeline complet AutoML.
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
import os
import tempfile

print("=" * 70)
print("TEST D'INTÉGRATION - MODULE MODELS")
print("=" * 70)

# Test 1: Import du module models directement
print("\n[1/5] Test import du module models...")
try:
    from automl.models import (
        BaseModel,
        ModelFactory,
        ModelTrainer,
        ModelSelector,
        train_models,
        get_best_model,
        get_trained_models,
        reset_models
    )
    print("✅ Tous les imports fonctionnent")
except Exception as e:
    print(f"❌ Erreur d'import: {e}")
    exit(1)

# Test 2: Création de modèles
print("\n[2/5] Test création de modèles...")
try:
    clf_models = ModelFactory.get_default_models('classification')
    reg_models = ModelFactory.get_default_models('regression')
    print(f"✅ {len(clf_models)} modèles de classification créés")
    print(f"✅ {len(reg_models)} modèles de régression créés")
except Exception as e:
    print(f"❌ Erreur création: {e}")
    exit(1)

# Test 3: Entraînement via train_models (interface core.py)
print("\n[3/5] Test entraînement via train_models()...")
try:
    # Générer des données
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_classes=2,
        random_state=42
    )
    X_train, X_valid = X[:200], X[200:]
    y_train, y_valid = y[:200], y[200:]

    # Appeler train_models comme le ferait core.py
    trained = train_models(
        X_train, y_train,
        X_valid, y_valid,
        task_type='classification',
        verbose=False,
        random_state=42
    )

    print(f"✅ {len(trained)} modèles entraînés")
    print(f"✅ Modèles disponibles: {list(trained.keys())}")
except Exception as e:
    print(f"❌ Erreur entraînement: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 4: Récupération du meilleur modèle
print("\n[4/5] Test récupération du meilleur modèle...")
try:
    best = get_best_model()
    print(f"✅ Meilleur modèle: {best.name}")
    print(f"✅ Score validation: {best.valid_score:.4f}")

    # Tester la prédiction
    predictions = best.predict(X_valid)
    print(f"✅ Prédictions: {len(predictions)} résultats")
except Exception as e:
    print(f"❌ Erreur récupération: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Sauvegarde et chargement
print("\n[5/5] Test sauvegarde et chargement...")
try:
    with tempfile.TemporaryDirectory() as tmpdir:
        # Sauvegarder
        path = best.save(tmpdir)
        print(f"✅ Modèle sauvegardé: {os.path.basename(path)}")

        # Charger
        loaded = BaseModel.load(path)
        print(f"✅ Modèle chargé: {loaded.name}")

        # Vérifier que les prédictions sont identiques
        pred_original = best.predict(X_valid)
        pred_loaded = loaded.predict(X_valid)

        if np.array_equal(pred_original, pred_loaded):
            print("✅ Prédictions identiques après chargement")
        else:
            print("❌ Les prédictions diffèrent")
            exit(1)
except Exception as e:
    print(f"❌ Erreur sauvegarde/chargement: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 6: Reset pour nettoyage
print("\n[6/6] Test reset du module...")
try:
    reset_models()
    print("✅ Module réinitialisé")
except Exception as e:
    print(f"❌ Erreur reset: {e}")
    exit(1)

# Récapitulatif
print("\n" + "=" * 70)
print("TOUS LES TESTS D'INTÉGRATION SONT RÉUSSIS ! 🎉")
print("=" * 70)
print("\nLe module models est prêt à être utilisé dans le pipeline AutoML.")
print("\nVérifications effectuées:")
print("  ✅ Imports corrects")
print("  ✅ Création de modèles")
print("  ✅ Entraînement via interface core.py")
print("  ✅ Récupération du meilleur modèle")
print("  ✅ Prédictions fonctionnelles")
print("  ✅ Sauvegarde/chargement")
print("  ✅ Reset du module")
print("\nProchain test: Intégration complète avec automl.fit()")
