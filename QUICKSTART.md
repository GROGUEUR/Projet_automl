# Quick Start - AutoML

Guide de démarrage rapide pour utiliser et tester le système AutoML.

## Installation

### 1. Cloner le projet

```bash
cd /path/to/Projet_automl
```

### 2. Installer le paquet

```bash
pip install -e .
```

Cette commande installe le paquet en mode éditable (les modifications du code sont immédiatement prises en compte).

## Test Rapide

### Vérifier l'installation

```bash
python -c "import automl; print(automl.__version__)"
```

Sortie attendue : `0.1.0`

### Exécuter l'exemple de démonstration

```bash
python example.py
```

Cet exemple :
1. Crée un dataset synthétique
2. Charge et prétraite les données
3. Affiche les informations sur les splits
4. Démontre l'interface complète

## Tests Unitaires

### Exécuter tous les tests

```bash
pytest tests/ -v
```

### Exécuter avec couverture de code

```bash
pytest --cov=automl tests/
```

### Tests spécifiques

```bash
# Tester uniquement le data loader
pytest tests/test_data_loader.py::TestDataLoader -v

# Tester le preprocessing
pytest tests/test_data_loader.py::TestDataPreprocessor -v

# Tester les splits
pytest tests/test_data_loader.py::TestTrainValidTestSplit -v
```

## Utilisation de Base

### Exemple 1 : Interface Minimale

```python
import automl

# Charger et entraîner en une ligne
automl.fit(data_path="/path/to/data.csv")

# Évaluer
automl.eval()
```

### Exemple 2 : Avec Paramètres Personnalisés

```python
import automl

automl.fit(
    data_path="/path/to/data.csv",
    train_size=0.6,
    valid_size=0.2,
    test_size=0.2,
    handle_missing='median',
    scale=True,
    verbose=True
)
```

### Exemple 3 : Accéder aux Données

```python
import automl

automl.fit(data_path="/path/to/data.csv")

# Récupérer les données
data = automl.get_data()

print(f"X_train shape: {data['X_train'].shape}")
print(f"X_valid shape: {data['X_valid'].shape}")
print(f"X_test shape: {data['X_test'].shape}")
print(f"Task type: {data['task_type']}")
```

## Utilisation des Modules Individuels

### Module DataLoader

```python
from automl.data import DataLoader

# Charger des données
loader = DataLoader("/path/to/data.csv")
X, y, task_type = loader.load_data()

# Obtenir des infos
info = loader.get_info()
print(info)
```

### Module DataPreprocessor

```python
from automl.data import DataPreprocessor

# Créer et utiliser le preprocessor
preprocessor = DataPreprocessor(
    handle_missing='mean',
    scale=True,
    encode_categorical=True
)

X_transformed = preprocessor.fit_transform(X)

# Sauvegarder le preprocessor
preprocessor.save('./saved_models')
```

### Fonction de Split

```python
from automl.data import train_valid_test_split

# Split les données
splits = train_valid_test_split(
    X, y,
    train_size=0.7,
    valid_size=0.15,
    test_size=0.15,
    task_type='classification'
)

X_train = splits['X_train']
X_valid = splits['X_valid']
X_test = splits['X_test']
```

## Configuration

### Afficher la Configuration

```python
from automl.utils import Config

Config.display()
```

### Modifier la Configuration

```python
from automl.utils import Config

# Modifier les paramètres
Config.TRAIN_SIZE = 0.8
Config.RANDOM_STATE = 123
Config.HANDLE_MISSING = 'median'

# Créer les répertoires nécessaires
Config.create_directories()
```

## Formats de Données Supportés

### CSV

```csv
1.0,2.0,3.0,0
4.0,5.0,6.0,1
7.0,8.0,9.0,0
```

### TXT (espace ou tabulation)

```
1.0 2.0 3.0 0
4.0 5.0 6.0 1
7.0 8.0 9.0 0
```

**Important :** La dernière colonne est toujours considérée comme la variable cible.

## Détection Automatique

Le système détecte automatiquement :

1. **Format du fichier** : CSV, TXT, DAT
2. **Séparateur** : virgule, point-virgule, espace, tabulation
3. **Type de tâche** :
   - Classification : < 20 valeurs uniques et < 5% du total
   - Régression : beaucoup de valeurs différentes
4. **Types de colonnes** : numériques vs catégorielles

## Dépannage

### Erreur : "ModuleNotFoundError: No module named 'automl'"

```bash
# Réinstaller le paquet
pip install -e .
```

### Erreur : "FileNotFoundError"

Vérifiez que le chemin vers vos données est correct :

```python
import os
print(os.path.exists("/path/to/data.csv"))  # Devrait afficher True
```

### Tests échouent

```bash
# Installer les dépendances de développement
pip install pytest pytest-cov

# Réexécuter les tests
pytest tests/ -v
```

### Problèmes avec les valeurs manquantes

Le système gère automatiquement les valeurs manquantes. Pour désactiver :

```python
automl.fit(data_path="...", handle_missing='drop')
```

## Prochaines Étapes

Une fois que l'infrastructure est validée :

1. **Personne 2** : Implémenter le module `models/`
2. **Personne 3** : Implémenter le module `optimization/`
3. **Personne 4** : Implémenter le module `evaluation/`

Voir [CONTRIBUTING.md](CONTRIBUTING.md) pour les détails d'intégration.

## Ressources

- [README.md](README.md) - Documentation complète
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guide de contribution
- [example.py](example.py) - Script de démonstration
- [tests/](tests/) - Tests unitaires

## Support

Pour toute question :
- **Infrastructure & Data** : Personne 1 (ce module)
- **Documentation** : README.md et CONTRIBUTING.md
- **Bugs** : Créer une issue sur le dépôt

## Validation de l'Installation

Pour valider que tout fonctionne correctement :

```bash
# 1. Vérifier l'import
python -c "import automl; print('OK')"

# 2. Exécuter les tests
pytest tests/ -v

# 3. Exécuter l'exemple
python example.py
```

Si ces trois commandes réussissent, l'installation est correcte.
