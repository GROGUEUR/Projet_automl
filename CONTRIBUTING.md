# Guide de Contribution - AutoML

Ce document explique comment intégrer votre module dans le système AutoML.

## Structure Générale

Le projet est organisé en modules indépendants qui communiquent via l'interface définie dans `automl/core.py`.

```
automl/
├── data/          # ✅ COMPLÉTÉ (Personne 1)
├── models/        # 🔜 À faire (Personne 2)
├── optimization/  # 🔜 À faire (Personne 3)
└── evaluation/    # 🔜 À faire (Personne 4)
```

## Pour Personne 2 : Module Models

### Objectif
Implémenter l'entraînement de plusieurs modèles sklearn.

### Fichiers à créer
- `automl/models/trainer.py`

### Interface attendue

```python
# automl/models/trainer.py
def train_models(X_train, y_train, X_valid, y_valid, task_type, **kwargs):
    """
    Entraîne plusieurs modèles sklearn.

    Args:
        X_train: Features d'entraînement (numpy array)
        y_train: Target d'entraînement (numpy array)
        X_valid: Features de validation (numpy array)
        y_valid: Target de validation (numpy array)
        task_type: 'classification' ou 'regression'
        **kwargs: Arguments supplémentaires (verbose, etc.)

    Returns:
        dict: Dictionnaire {nom_modèle: modèle_entraîné}
    """
    trained_models = {}

    # Votre code ici
    # Exemple pour classification:
    if task_type == 'classification':
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier

        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        trained_models['logistic_regression'] = lr

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        trained_models['random_forest'] = rf

    return trained_models
```

### Mise à jour de __init__.py

```python
# automl/models/__init__.py
from .trainer import train_models

__all__ = ['train_models']
```

### Comment tester

```python
import automl

# Charger et préparer les données
automl.fit(data_path="/path/to/data")

# Vos modèles seront automatiquement entraînés
data = automl.get_data()
print(data['trained_models'])
```

## Pour Personne 3 : Module Optimization

### Objectif
Optimiser les hyperparamètres des modèles entraînés.

### Fichiers à créer
- `automl/optimization/optimizer.py`

### Interface attendue

```python
# automl/optimization/optimizer.py
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

def optimize_hyperparameters(model, X_train, y_train, param_grid, **kwargs):
    """
    Optimise les hyperparamètres d'un modèle.

    Args:
        model: Modèle sklearn à optimiser
        X_train: Features d'entraînement
        y_train: Target d'entraînement
        param_grid: Grille de paramètres
        **kwargs: cv, n_iter, scoring, etc.

    Returns:
        model: Modèle avec les meilleurs paramètres
        dict: Meilleurs paramètres trouvés
    """
    cv = kwargs.get('cv', 5)
    n_iter = kwargs.get('n_iter', 20)

    search = RandomizedSearchCV(
        model,
        param_grid,
        n_iter=n_iter,
        cv=cv,
        random_state=42
    )
    search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
```

### Intégration avec le module models

Le module optimization peut être appelé depuis le module models :

```python
# Dans models/trainer.py
from automl.optimization import optimize_hyperparameters

def train_models(...):
    # Entraîner le modèle de base
    model = RandomForestClassifier()

    # Définir la grille de paramètres
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10, None]
    }

    # Optimiser
    optimized_model, best_params = optimize_hyperparameters(
        model, X_train, y_train, param_grid
    )

    return optimized_model
```

## Pour Personne 4 : Module Evaluation

### Objectif
Évaluer les modèles entraînés avec différentes métriques.

### Fichiers à créer
- `automl/evaluation/evaluator.py`

### Interface attendue

```python
# automl/evaluation/evaluator.py
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(trained_models, X_test, y_test, task_type, **kwargs):
    """
    Évalue tous les modèles entraînés.

    Args:
        trained_models: Dict {nom: modèle}
        X_test: Features de test
        y_test: Target de test
        task_type: 'classification' ou 'regression'
        **kwargs: verbose, save_results, etc.

    Returns:
        dict: Résultats d'évaluation pour chaque modèle
    """
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        if task_type == 'classification':
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
        else:
            # Métriques de régression
            pass

    return results
```

### Mise à jour de __init__.py

```python
# automl/evaluation/__init__.py
from .evaluator import evaluate_models

__all__ = ['evaluate_models']
```

### Comment tester

```python
import automl

automl.fit(data_path="/path/to/data")
results = automl.eval()  # Appelle votre module automatiquement
print(results)
```

## Accès aux Données

Tous les modules peuvent accéder aux données via `get_data()` :

```python
from automl.core import get_data

def your_function():
    data = get_data()
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    X_test = data['X_test']
    y_test = data['y_test']
    task_type = data['task_type']
    trained_models = data['trained_models']

    # Votre code ici
```

## Configuration

Utilisez `Config` pour accéder aux paramètres :

```python
from automl.utils import Config

# Lire les paramètres
n_jobs = Config.N_JOBS
random_state = Config.RANDOM_STATE
verbose = Config.VERBOSE

# Afficher la config
Config.display()
```

## Tests

Créez des tests pour votre module :

```python
# tests/test_your_module.py
import pytest
from automl.your_module import your_function

def test_your_function():
    # Votre test ici
    result = your_function(...)
    assert result is not None
```

Exécuter les tests :

```bash
pytest tests/test_your_module.py -v
```

## Convention de Code

1. **Style PEP8**
   ```bash
   flake8 automl/your_module/
   ```

2. **Docstrings** (format Google)
   ```python
   def function(arg1, arg2):
       """
       Description courte.

       Description longue si nécessaire.

       Args:
           arg1: Description de arg1
           arg2: Description de arg2

       Returns:
           Description du retour

       Raises:
           ValueError: Quand arg1 est invalide
       """
   ```

3. **Type hints**
   ```python
   from typing import Dict, List, Optional

   def function(x: np.ndarray, y: Optional[str] = None) -> Dict[str, float]:
       pass
   ```

## Workflow Git

1. **Créer une branche pour votre module**
   ```bash
   git checkout -b feature/models  # ou optimization, ou evaluation
   ```

2. **Faire vos commits**
   ```bash
   git add automl/your_module/
   git commit -m "Add: Module your_module implementation"
   ```

3. **Tester avant de push**
   ```bash
   pytest tests/
   ```

4. **Push et créer une Pull Request**
   ```bash
   git push origin feature/your_module
   ```

## Points d'Attention

### Types de Données
- X toujours en numpy array, shape `(n_samples, n_features)`
- y toujours en numpy array, shape `(n_samples,)` (1D)
- task_type exactement `'classification'` ou `'regression'`

### Nommage
- Respecter exactement : `X_train`, `X_valid`, `X_test`, `y_train`, `y_valid`, `y_test`
- Pas de `X_val`, pas de `X_validation`

### Gestion d'Erreurs
```python
if condition_invalide:
    raise ValueError("Message clair et explicite")
```

### Logging
```python
if verbose:
    print("Information utile pour l'utilisateur")
```

## Intégration Continue

Une fois votre module terminé :

1. ✅ Tests unitaires passent
2. ✅ Code respecte PEP8
3. ✅ Docstrings complètes
4. ✅ Exemple d'utilisation dans README
5. ✅ Pull Request créée

## Besoin d'Aide ?

- **Questions sur l'infrastructure** : Voir Personne 1
- **Questions sur les données** : Voir [automl/data/](automl/data/)
- **Configuration** : Voir [automl/utils/config.py](automl/utils/config.py)

## Exemple Complet d'Intégration

Voir [example.py](example.py) pour un exemple complet d'utilisation du système.
