# AutoML - Système d'Apprentissage Automatique Automatisé

Projet M1 Informatique IA - Pipeline automatisé d'entraînement et d'évaluation de modèles de machine learning.

## Description

AutoML est un système complet qui automatise le processus d'apprentissage automatique de bout en bout :

- Chargement et préparation des données (multi-formats)
- Détection automatique du type de tâche (classification/régression)
- Prétraitement intelligent (valeurs manquantes, normalisation, encodage)
- Entraînement de multiples modèles sklearn (13 algorithmes)
- Sélection automatique du meilleur modèle
- Optimisation des hyperparamètres (Grid/Random Search)
- Évaluation complète avec métriques et visualisations

**Caractéristiques principales:**

- Interface simple : 2 lignes de code suffisent (`fit` + `eval`)
- 7 modèles de classification et 6 de régression
- Espaces d'hyperparamètres prédéfinis
- Visualisations automatiques (ROC, confusion matrix, résidus)
- Support du format ChallengeMachineLearning
- Architecture modulaire et extensible

## Installation

```bash
# Cloner le dépôt
cd /path/to/Projet_automl

# Installer les dépendances
pip install -e .
pip install -r requirements.txt
```

## Formats de Données Supportés

Le système accepte plusieurs formats :

**Formats de fichiers:**

- CSV (`.csv`) avec séparateur automatique (`,`, `;`, `\t`, espace)
- Fichiers texte (`.txt`)
- Format ChallengeMachineLearning (`.data` + `.solution`)

**Structure attendue:**

- Dernière colonne = variable cible (y)
- Autres colonnes = features (X)
- En-têtes optionnels

## Utilisation

### Script d'Exemple

Un script de démonstration complet est disponible :

```bash
python example.py
```

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
├── automl/                          # Package principal
│   ├── __init__.py                 # Interface publique (fit, eval, get_data, reset)
│   ├── core.py                     # Orchestration du pipeline
│   ├── data/                       # Module de gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py               # DataLoader - Chargement des données
│   │   └── preprocessing.py        # DataPreprocessor - Prétraitement et splits
│   ├── models/                     # Module d'entraînement
│   │   ├── __init__.py
│   │   ├── base_model.py           # BaseModel - Wrapper pour sklearn
│   │   ├── model_factory.py        # ModelFactory - Création de modèles
│   │   ├── model_trainer.py        # ModelTrainer - Orchestration
│   │   └── model_selector.py       # ModelSelector - Sélection automatique
│   ├── optimization/               # Module d'optimisation
│   │   ├── __init__.py
│   │   ├── hyperparameter_space.py # Espaces de paramètres
│   │   ├── hyparparameter_tuner.py # Grid/Random Search
│   │   └── optimization_pipeline.py # Pipeline d'optimisation
│   ├── evaluation/                 # Module d'évaluation
│   │   ├── __init__.py
│   │   ├── evaluator.py            # ModelEvaluator - Évaluation
│   │   ├── metrics.py              # MetricsCalculator - Métriques
│   │   └── visualizer.py           # ResultsVisualizer - Graphiques
│   └── utils/                      # Utilitaires
│       ├── __init__.py
│       └── config.py               # Configuration globale
├── setup.py                        # Configuration d'installation
├── requirements.txt                # Dépendances
├── example.py                      # Script d'exemple
└── README.md                       # Documentation
```

## Modules Détaillés

### 1. Infrastructure & Gestion des Données

**Responsable:** Chargement, prétraitement et organisation des données

**Fichiers:**

- [automl/data/loader.py](automl/data/loader.py) - Classe `DataLoader`
- [automl/data/preprocessing.py](automl/data/preprocessing.py) - Classe `DataPreprocessor`
- [automl/utils/config.py](automl/utils/config.py) - Configuration globale
- [automl/core.py](automl/core.py) - Interface principale

**Fonctionnalités:**

- Chargement automatique de fichiers CSV, TXT, .data/.solution
- Détection automatique du séparateur
- Support du format ChallengeMachineLearning (fichier.data + fichier.solution)
- Détection du type de tâche (classification/régression)
- Gestion des valeurs manquantes (mean, median, most_frequent, drop)
- Normalisation des features numériques (StandardScaler)
- Encodage des variables catégorielles (LabelEncoder)
- Split train/valid/test avec stratification pour classification
- Sauvegarde des preprocessors (joblib)

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

### 2. Entraînement des Modèles

**Responsable:** Entraînement et sélection de multiples modèles

**Fichiers:**

- [automl/models/base_model.py](automl/models/base_model.py) - Classe `BaseModel`
- [automl/models/model_factory.py](automl/models/model_factory.py) - Classe `ModelFactory`
- [automl/models/model_trainer.py](automl/models/model_trainer.py) - Classe `ModelTrainer`
- [automl/models/model_selector.py](automl/models/model_selector.py) - Classe `ModelSelector`

**Fonctionnalités:**

- Wrapper unifié pour modèles sklearn (BaseModel)
- Factory pour créer des modèles par type de tâche
- Entraînement parallèle ou séquentiel de plusieurs modèles
- Sélection automatique du meilleur modèle
- Sérialisation avec joblib

**Modèles supportés:**

**Classification (7):**

- RandomForest, GradientBoosting, LogisticRegression
- SVM, KNN, DecisionTree, NaiveBayes

**Régression (6):**

- RandomForest, GradientBoosting, Ridge
- SVR, KNN, DecisionTree

**API ModelTrainer:**

```python
from automl.models import ModelTrainer

trainer = ModelTrainer()
results = trainer.train_models(
    X_train, y_train,
    X_valid, y_valid,
    task_type='classification'
)
```

**API ModelSelector:**

```python
from automl.models import ModelSelector

selector = ModelSelector(trained_models, X_valid, y_valid)

# Meilleur modèle par score
best = selector.select_by_score()

# Meilleur rapport vitesse/performance
best = selector.select_by_speed_score_tradeoff()

# Top k modèles
top_models = selector.select_top_k(k=3)
```

### 3. Optimisation des Hyperparamètres

**Responsable:** Optimisation des hyperparamètres des modèles

**Fichiers:**

- [automl/optimization/hyperparameter_space.py](automl/optimization/hyperparameter_space.py) - Classe `HyperparameterSpace`
- [automl/optimization/hyparparameter_tuner.py](automl/optimization/hyparparameter_tuner.py) - Classe `HyperparameterTuner`
- [automl/optimization/optimization_pipeline.py](automl/optimization/optimization_pipeline.py) - Classe `OptimizationPipeline`

**Fonctionnalités:**

- Espaces de paramètres prédéfinis pour tous les modèles
- Grid Search et Random Search
- Validation croisée configurable
- Pipeline d'optimisation complet avec comparaison avant/après

**API HyperparameterTuner:**

```python
from automl.optimization import HyperparameterTuner

tuner = HyperparameterTuner(method='grid', cv=5)
best_params = tuner.optimize(
    model, X_train, y_train,
    param_grid={...}
)
```

### 4. Évaluation

**Responsable:** Évaluation et visualisation des performances

**Fichiers:**

- [automl/evaluation/evaluator.py](automl/evaluation/evaluator.py) - Classe `ModelEvaluator`
- [automl/evaluation/metrics.py](automl/evaluation/metrics.py) - Classe `MetricsCalculator`
- [automl/evaluation/visualizer.py](automl/evaluation/visualizer.py) - Classe `ResultsVisualizer`

**Fonctionnalités:**

- Évaluation simple ou multiple de modèles
- Calcul de métriques complètes par type de tâche
- Matrices de confusion et rapports de classification
- Visualisations (courbes ROC, résidus, comparaisons)

**Métriques:**

**Classification:**

- Accuracy, Precision, Recall, F1-score, ROC-AUC

**Régression:**

- R², RMSE, MAE, MAPE, Max Error

**API ModelEvaluator:**

```python
from automl.evaluation import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate_models(
    trained_models,
    X_test, y_test,
    task_type='classification'
)
```

**API ResultsVisualizer:**

```python
from automl.evaluation import ResultsVisualizer

visualizer = ResultsVisualizer()

# Comparaison des modèles
visualizer.plot_model_comparison(results, save_path='comparison.png')

# Matrice de confusion
visualizer.plot_confusion_matrix(y_true, y_pred, save_path='confusion.png')

# Courbe ROC (classification binaire)
visualizer.plot_roc_curve(y_true, y_pred_proba, save_path='roc.png')

# Analyse des résidus (régression)
visualizer.plot_residuals(y_true, y_pred, save_path='residuals.png')
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
- `SAVE_DIR` : Répertoire de sauvegarde des modèles
- `TRAIN_SIZE`, `VALID_SIZE`, `TEST_SIZE` : Proportions des splits (défaut: 0.7, 0.15, 0.15)
- `HANDLE_MISSING` : Stratégie pour valeurs manquantes ('mean', 'median', 'most_frequent', 'drop')
- `SCALE_FEATURES` : Normalisation des features (défaut: True)
- `ENCODE_CATEGORICAL` : Encodage catégoriel (défaut: True)
- `RANDOM_STATE` : Graine aléatoire pour reproductibilité (défaut: 42)
- `N_JOBS` : Nombre de processus parallèles (défaut: -1 pour tous les CPUs)
- `VERBOSE` : Mode verbeux (défaut: False)
- `CV_FOLDS` : Nombre de folds pour validation croisée (défaut: 5)
- `OPTIMIZATION_N_ITER` : Nombre d'itérations pour RandomSearch (défaut: 20)

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

### Modèles

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

### Optimisation

```python
# Utiliser les données prétraitées
from automl.core import get_data

data = get_data()
X_train = data['X_train']
y_train = data['y_train']
task_type = data['task_type']
```

### Évaluation

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

**Dépendances principales:**

- **numpy** >= 1.24.3 : Calcul numérique
- **pandas** >= 2.0.3 : Manipulation de données
- **scikit-learn** >= 1.3.0 : Algorithmes ML
- **joblib** >= 1.3.2 : Sérialisation
- **matplotlib** >= 3.7.0 : Visualisation
- **seaborn** >= 0.12.0 : Graphiques statistiques

**Dépendances de développement:**

- **pytest** >= 7.4.0 : Tests unitaires
- **pytest-cov** >= 4.1.0 : Couverture de code
- **flake8** >= 6.0.0 : Linting
- **black** >= 23.7.0 : Formatage de code

### Convention de Code

- **Style:** PEP8
- **Docstrings:** Format Google
- **Type hints:** Obligatoires pour les fonctions publiques
