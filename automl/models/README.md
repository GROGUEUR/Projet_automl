# Module Models - Sélection & Entraînement des Modèles

**Responsable:** Personne 2
**Statut:** ✅ Complet et testé

## Description

Ce module gère la sélection, l'entraînement et la comparaison automatique de modèles sklearn pour des tâches de classification et régression.

## Architecture

### Fichiers

```
automl/models/
├── __init__.py           # Exports et fonctions d'intégration
├── base_model.py         # Classe BaseModel encapsulant sklearn
├── model_factory.py      # Création des modèles selon le type de tâche
├── model_trainer.py      # Entraînement et comparaison de modèles
├── model_selector.py     # Stratégies de sélection avancées
└── README.md            # Cette documentation
```

## Classes Principales

### 1. BaseModel

Encapsule un modèle sklearn avec des métadonnées.

```python
from automl.models import BaseModel
from sklearn.ensemble import RandomForestClassifier

# Créer un modèle
model = BaseModel(
    name="MonModele",
    model=RandomForestClassifier(random_state=42),
    task_type='classification'
)

# Entraîner
model.fit(X_train, y_train, X_valid, y_valid)

# Prédire
predictions = model.predict(X_test)

# Sauvegarder
model.save("./saved_models")

# Charger
loaded = BaseModel.load("./saved_models/MonModele_20241217_123456.joblib")
```

**Attributs:**
- `name`: nom du modèle
- `task_type`: 'classification' ou 'regression'
- `is_fitted`: booléen indiquant si le modèle est entraîné
- `train_score`: score sur l'ensemble d'entraînement
- `valid_score`: score sur l'ensemble de validation
- `training_time`: temps d'entraînement en secondes

### 2. ModelFactory

Crée automatiquement des modèles selon le type de tâche.

```python
from automl.models import ModelFactory

# Obtenir tous les modèles par défaut
models = ModelFactory.get_default_models('classification')
# Retourne 7 modèles: RandomForest, GradientBoosting, LogisticRegression,
#                     SVM, KNN, DecisionTree, NaiveBayes

models = ModelFactory.get_default_models('regression')
# Retourne 6 modèles: RandomForest, GradientBoosting, Ridge,
#                     SVR, KNN, DecisionTree

# Créer un modèle spécifique avec paramètres personnalisés
rf = ModelFactory.create_model(
    'RandomForest',
    'classification',
    n_estimators=200,
    max_depth=10
)

# Lister les modèles disponibles
available = ModelFactory.get_available_models('classification')
```

**Modèles disponibles:**

| Classification | Régression |
|---------------|------------|
| RandomForest | RandomForest |
| GradientBoosting | GradientBoosting |
| LogisticRegression | Ridge |
| SVM | SVR |
| KNN | KNN |
| DecisionTree | DecisionTree |
| NaiveBayes | - |

### 3. ModelTrainer

Entraîne et compare automatiquement plusieurs modèles.

```python
from automl.models import ModelTrainer

# Créer un trainer
trainer = ModelTrainer(
    task_type='classification',
    random_state=42,
    verbose=True
)

# Entraîner tous les modèles
results = trainer.train_all(X_train, y_train, X_valid, y_valid)

# Sélectionner le meilleur
best_model = trainer.select_best_model(metric='valid_score')

# Obtenir le résumé
summary = trainer.get_results_summary()
print(summary)

# Sauvegarder tous les modèles
trainer.save_all_models("./saved_models")

# Ou seulement le meilleur
trainer.save_best_model("./saved_models")

# Récupérer un modèle spécifique
rf_model = trainer.get_model('RandomForest')
```

**Résultats retournés:**
```python
[
    {
        'name': 'RandomForest',
        'train_score': 0.98,
        'valid_score': 0.92,
        'training_time': 0.15
    },
    ...
]
```

### 4. ModelSelector

Stratégies avancées de sélection de modèles.

```python
from automl.models import ModelSelector

# Sélection par meilleur score
best = ModelSelector.select_by_score(results, metric='valid_score')

# Compromis vitesse/performance
best = ModelSelector.select_by_speed_score_tradeoff(
    results,
    score_weight=0.7,
    speed_weight=0.3
)

# Top K modèles
top_3 = ModelSelector.select_top_k(results, k=3)

# Contrôle du surapprentissage
best = ModelSelector.select_by_overfitting_control(results, max_gap=0.1)

# Classement complet
rankings = ModelSelector.get_model_rankings(results)
```

## Utilisation dans AutoML

Le module s'intègre automatiquement dans le pipeline AutoML:

```python
import automl

# Le module models est appelé automatiquement par fit()
automl.fit(data_path="/path/to/data")

# Les modèles sont accessibles via:
from automl.models import get_best_model, get_trained_models

best = get_best_model()
all_models = get_trained_models()
```

## Métriques d'Évaluation

- **Classification:** accuracy par défaut
- **Régression:** R² par défaut

Ces métriques seront enrichies par le module d'évaluation (Personne 4).

## Tests

Exécuter les tests unitaires:

```bash
pytest tests/test_models.py -v
```

Tous les tests passent (23/23) ✅

## Exemple Complet

```python
from automl.models import ModelTrainer, ModelSelector
from sklearn.datasets import make_classification

# Générer des données
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_valid = X[:800], X[800:]
y_train, y_valid = y[:800], y[800:]

# Entraîner tous les modèles
trainer = ModelTrainer(task_type='classification', verbose=True)
results = trainer.train_all(X_train, y_train, X_valid, y_valid)

# Afficher le résumé
print(trainer.get_results_summary())

# Sélectionner le meilleur
best = trainer.select_best_model()

# Utiliser pour prédire
predictions = best.predict(X_valid)

# Sauvegarder
trainer.save_best_model("./models")
```

## Points d'Intégration avec les Autres Modules

### Avec Personne 1 (Data)
- **Input:** Reçoit `X_train, y_train, X_valid, y_valid, task_type` depuis `core.py`
- **Utilise:** Les données prétraitées et splitées

### Avec Personne 3 (Optimisation)
- **Fournit:** Accès aux modèles via `get_params()` et `set_params()`
- **Interface:** Les modèles sklearn peuvent être optimisés directement

### Avec Personne 4 (Évaluation)
- **Fournit:** Modèles entraînés via `get_best_model()` ou `get_trained_models()`
- **Interface:** Les modèles peuvent prédire sur les données de test

## Configuration

Les paramètres par défaut sont dans `automl/utils/config.py`:

```python
class Config:
    RANDOM_STATE = 42
    N_JOBS = -1  # Parallélisation
    MODELS_SAVE_PATH = "./saved_models"
```

## Reproductibilité

- Tous les modèles utilisent `random_state=42` par défaut
- Les résultats sont déterministes
- Les modèles peuvent être sauvegardés et rechargés

## Performances

Le module a été testé sur:
- ✅ Datasets de classification (2+ classes)
- ✅ Datasets de régression
- ✅ Gestion des modèles échouant gracieusement
- ✅ Sauvegarde/chargement des modèles
- ✅ Sélection du meilleur modèle

## Notes Techniques

1. **Parallélisation:** Les modèles supportant `n_jobs=-1` l'utilisent automatiquement
2. **Probabilités:** Pour classification, `SVM` a `probability=True` pour `predict_proba()`
3. **Timeout:** Aucun timeout sur l'entraînement (peut être long pour SVM)
4. **Mémoire:** Les modèles sont gardés en mémoire jusqu'à `reset_models()`

## Auteur

Implémenté par Personne 2 (Bastien DELAMARE) dans le cadre du projet AutoML M1 Info IA.

## Statut

✅ **Complet et prêt pour l'intégration**

Tous les livrables attendus sont réalisés:
- ✅ BaseModel avec métadonnées
- ✅ ModelFactory avec 7 modèles classification + 6 régression
- ✅ ModelTrainer fonctionnel
- ✅ ModelSelector avec 4+ stratégies
- ✅ Intégration dans core.py
- ✅ Tests unitaires (23/23 passent)
- ✅ Documentation complète
