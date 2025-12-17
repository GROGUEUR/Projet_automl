# AutoML - Système d'Apprentissage Automatique Automatisé

Projet M1 Informatique IA - Pipeline automatisé d'entraînement et d'évaluation de modèles de machine learning.

## Description

AutoML est un système complet qui automatise le processus d'apprentissage automatique de bout en bout :
- Chargement et préparation des données
- Détection automatique du type de tâche (classification/régression)
- Prétraitement intelligent des données
- Entraînement de multiples modèles sklearn
- Optimisation des hyperparamètres
- Évaluation des performances

## Installation

### Méthode 1 : Installation en mode développement (recommandée)

```bash
# Cloner le dépôt
cd /path/to/Projet_automl

# Installer le paquet en mode éditable
pip install -e .
```

### Méthode 2 : Installation avec requirements.txt

```bash
pip install -r requirements.txt
```

## Utilisation

### Interface Minimale

L'interface utilisateur est volontairement simple et intuitive :

```python
import automl

# Charger et entraîner
automl.fit(data_path="/path/to/data")

# Évaluer les modèles
automl.eval()
```

### Exemple Complet

```python
import automl

# Entraînement avec paramètres personnalisés
automl.fit(
    data_path="/info/corpus/ChallengeMachineLearning/dataset1",
    train_size=0.7,
    valid_size=0.15,
    test_size=0.15,
    handle_missing='mean',
    scale=True,
    encode_categorical=True,
    verbose=True
)

# Évaluation
results = automl.eval(verbose=True)

# Accéder aux données (pour debugging)
data = automl.get_data()
print(f"Shape de X_train: {data['X_train'].shape}")
print(f"Type de tâche: {data['task_type']}")
```

## Structure du Projet

```
Projet_automl/
├── automl/                    # Package principal
│   ├── __init__.py           # Interface publique (fit, eval, get_data)
│   ├── core.py               # Orchestration du pipeline
│   ├── data/                 # Module de gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py         # Chargement des données
│   │   └── preprocessing.py  # Prétraitement et splits
│   ├── models/               # Module d'entraînement (Personne 2)
│   │   └── __init__.py
│   ├── optimization/         # Module d'optimisation (Personne 3)
│   │   └── __init__.py
│   ├── evaluation/           # Module d'évaluation (Personne 4)
│   │   └── __init__.py
│   └── utils/                # Utilitaires
│       ├── __init__.py
│       └── config.py         # Configuration globale
├── tests/                     # Tests unitaires
│   └── test_data_loader.py
├── setup.py                   # Configuration d'installation
├── requirements.txt           # Dépendances
└── README.md                  # Documentation
```

## Modules Détaillés

### 1. Infrastructure & Gestion des Données (Personne 1) ✅

**Responsable:** Chargement, prétraitement et organisation des données

**Fichiers:**
- [automl/data/loader.py](automl/data/loader.py) - Classe `DataLoader`
- [automl/data/preprocessing.py](automl/data/preprocessing.py) - Classe `DataPreprocessor`
- [automl/utils/config.py](automl/utils/config.py) - Configuration globale
- [automl/core.py](automl/core.py) - Interface principale

**Fonctionnalités:**
- ✅ Chargement automatique de fichiers CSV, TXT
- ✅ Détection automatique du séparateur
- ✅ Détection du type de tâche (classification/régression)
- ✅ Gestion des valeurs manquantes (mean, median, most_frequent)
- ✅ Normalisation des features numériques (StandardScaler)
- ✅ Encodage des variables catégorielles (LabelEncoder)
- ✅ Split train/valid/test avec stratification
- ✅ Sauvegarde des preprocessors (joblib)

**API DataLoader:**

```python
from automl.data import DataLoader

loader = DataLoader(data_path="/path/to/data")
X, y, task_type = loader.load_data()
info = loader.get_info()
```

**API DataPreprocessor:**

```python
from automl.data import DataPreprocessor

preprocessor = DataPreprocessor(
    handle_missing='mean',
    scale=True,
    encode_categorical=True
)
X_transformed = preprocessor.fit_transform(X)
preprocessor.save('./saved_models')
```

**API Split:**

```python
from automl.data import train_valid_test_split

splits = train_valid_test_split(
    X, y,
    train_size=0.7,
    valid_size=0.15,
    test_size=0.15,
    task_type='classification'
)
X_train = splits['X_train']
```

### 2. Entraînement des Modèles (Personne 2) 🔜

**À implémenter:**
- Classe `ModelTrainer`
- Support de multiples algorithmes sklearn
- Entraînement parallèle des modèles
- Sauvegarde des modèles entraînés

**Interface attendue:**

```python
from automl.models import train_models

trained_models = train_models(
    X_train, y_train,
    X_valid, y_valid,
    task_type='classification'
)
```

### 3. Optimisation des Hyperparamètres (Personne 3) 🔜

**À implémenter:**
- Recherche d'hyperparamètres (Grid Search, Random Search)
- Validation croisée
- Optimisation par modèle

**Interface attendue:**

```python
from automl.optimization import optimize_hyperparameters

best_params = optimize_hyperparameters(
    model, X_train, y_train,
    param_grid={...}
)
```

### 4. Évaluation (Personne 4) 🔜

**À implémenter:**
- Calcul des métriques de performance
- Matrices de confusion
- Courbes ROC
- Rapports d'évaluation

**Interface attendue:**

```python
from automl.evaluation import evaluate_models

results = evaluate_models(
    trained_models,
    X_test, y_test,
    task_type='classification'
)
```

## Configuration

Le fichier [automl/utils/config.py](automl/utils/config.py) contient tous les paramètres par défaut :

```python
from automl.utils import Config

# Afficher la configuration
Config.display()

# Modifier les paramètres
Config.TRAIN_SIZE = 0.8
Config.RANDOM_STATE = 123
```

**Paramètres disponibles:**
- `DATA_PATH` : Chemin vers les données
- `TRAIN_SIZE`, `VALID_SIZE`, `TEST_SIZE` : Proportions des splits
- `HANDLE_MISSING` : Stratégie pour valeurs manquantes
- `SCALE_FEATURES` : Normalisation
- `ENCODE_CATEGORICAL` : Encodage catégoriel
- `RANDOM_STATE` : Reproductibilité

## Tests

### Exécution des Tests

```bash
# Tous les tests
pytest tests/

# Tests avec couverture
pytest --cov=automl tests/

# Tests spécifiques
pytest tests/test_data_loader.py
```

### Tests Disponibles

- ✅ Test de chargement CSV
- ✅ Test de détection du type de tâche
- ✅ Test du prétraitement
- ✅ Test du split train/valid/test
- ✅ Test de gestion des valeurs manquantes

## Formats de Données Supportés

### CSV
```
feature1,feature2,feature3,target
1.0,2.0,3.0,0
4.0,5.0,6.0,1
```

### TXT (séparateurs : espace, tabulation, virgule)
```
1.0 2.0 3.0 0
4.0 5.0 6.0 1
```

**Convention:** La dernière colonne est toujours la variable cible.

## Détection Automatique du Type de Tâche

Le système détecte automatiquement s'il s'agit de classification ou régression :

**Classification :**
- Type object/string dans la cible
- Moins de 20 valeurs uniques ET < 5% du total

**Régression :**
- Type numérique avec beaucoup de valeurs différentes

## Gestion des Valeurs Manquantes

Trois stratégies disponibles :

1. **'mean'** : Remplacement par la moyenne (numériques)
2. **'median'** : Remplacement par la médiane (numériques)
3. **'most_frequent'** : Remplacement par la valeur la plus fréquente
4. **'drop'** : Suppression des lignes (non recommandé)

## Points d'Intégration

### Pour Personne 2 (Modèles)

```python
# Dans automl/models/trainer.py
from automl.core import get_data

def train_models(X_train, y_train, X_valid, y_valid, task_type, **kwargs):
    # Accéder aux données
    data = get_data()

    # Entraîner vos modèles
    models = {}
    # ...

    return models
```

### Pour Personne 3 (Optimisation)

```python
# Utiliser les données prétraitées
from automl.core import get_data

data = get_data()
X_train = data['X_train']
y_train = data['y_train']
task_type = data['task_type']
```

### Pour Personne 4 (Évaluation)

```python
# Accéder aux modèles entraînés et données de test
from automl.core import get_data

data = get_data()
trained_models = data['trained_models']
X_test = data['X_test']
y_test = data['y_test']
```

## Reproductibilité

Le système garantit la reproductibilité via :
- Graine aléatoire fixe (`RANDOM_STATE = 42`)
- Versions fixes des dépendances (requirements.txt)
- Sauvegarde des preprocessors et modèles

## Dépendances

- **numpy** >= 1.21.0 : Calcul numérique
- **pandas** >= 1.3.0 : Manipulation de données
- **scikit-learn** >= 1.0.0 : Algorithmes ML
- **joblib** >= 1.0.0 : Sérialisation

## Développement

### Ajouter de nouvelles fonctionnalités

```bash
# Installation en mode développement
pip install -e ".[dev]"

# Formatter le code
black automl/

# Vérifier le style
flake8 automl/
```

### Convention de Code

- **Style:** PEP8
- **Docstrings:** Format Google
- **Type hints:** Obligatoires pour les fonctions publiques
- **Tests:** pytest pour tous les modules critiques

## Contribution

Chaque personne travaille sur son module :
1. **Personne 1** : Infrastructure & Data ✅
2. **Personne 2** : Entraînement des modèles 🔜
3. **Personne 3** : Optimisation des hyperparamètres 🔜
4. **Personne 4** : Évaluation 🔜

## Licence

MIT License - Projet académique M1 Info IA

## Contact

Pour toute question sur l'infrastructure et les données :
- Module data/ : Personne 1
- Module models/ : Personne 2
- Module optimization/ : Personne 3
- Module evaluation/ : Personne 4

## Statut du Projet

- [x] Infrastructure de base
- [x] Chargement des données
- [x] Prétraitement
- [x] Interface principale
- [ ] Entraînement des modèles
- [ ] Optimisation des hyperparamètres
- [ ] Évaluation
- [ ] Documentation complète

## Changelog

### Version 0.1.0 (Actuelle)
- Infrastructure de base complète
- Module de chargement des données
- Module de prétraitement
- Interface fit/eval/get_data
- Configuration centralisée
- Tests unitaires pour les données
- Documentation complète
