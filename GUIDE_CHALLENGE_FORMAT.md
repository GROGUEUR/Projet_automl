# Guide d'utilisation avec le format ChallengeMachineLearning

## Format des données

Le système AutoML supporte maintenant deux formats de données :

### Format 1 : Fichier unique (original)
Un seul fichier CSV/TXT avec features + target dans la dernière colonne.

```
feature1,feature2,feature3,target
1.0,2.5,A,0
2.0,3.1,B,1
```

### Format 2 : ChallengeMachineLearning (nouveau)
Structure de répertoire avec fichiers séparés :

```
ChallengeMachineLearning/
└── data_A/
    ├── data_A.data      # Features (séparées par espaces)
    ├── data_A.solution  # Target (une colonne)
    └── data_A.type      # Type de problème (optionnel)
```

## Utilisation avec le format Challenge

### Exemple simple

```python
import automl

# Pointer vers le répertoire du dataset
automl.fit(data_path="/info/corpus/ChallengeMachineLearning/data_A")

# Évaluer
automl.eval()
```

### Exemple complet

```python
import automl

# Configuration
DATASET = 'D'  # Changer la lettre ici
BASE_PATH = "/info/corpus/ChallengeMachineLearning"

# Chemin complet
data_path = f"{BASE_PATH}/data_{DATASET}"

# Entraînement
automl.fit(
    data_path=data_path,
    train_size=0.7,
    valid_size=0.15,
    test_size=0.15,
    handle_missing='mean',
    scale=True,
    verbose=True
)

# Évaluation
results = automl.eval(verbose=True)

# Accéder aux résultats
data = automl.get_data()
print(f"Type de tâche: {data['task_type']}")
print(f"Nombre de modèles: {len(data['trained_models'])}")
```

### Boucle sur plusieurs datasets

```python
import automl

BASE_PATH = "/info/corpus/ChallengeMachineLearning"

# Analyser les datasets A, B, C, D
for dataset_letter in ['A', 'B', 'C', 'D']:
    print(f"\n{'='*60}")
    print(f"DATASET {dataset_letter}")
    print(f"{'='*60}\n")

    # Réinitialiser pour chaque nouveau dataset
    automl.reset()

    # Chemin
    data_path = f"{BASE_PATH}/data_{dataset_letter}"

    try:
        # Entraîner
        automl.fit(data_path=data_path, verbose=True)

        # Évaluer
        results = automl.eval(verbose=True)

        # Récupérer le meilleur modèle
        from automl.models import get_best_model
        best = get_best_model(metric='valid_score')
        print(f"\n🏆 Meilleur modèle: {best.name}")
        print(f"   Score: {best.metadata['valid_score']:.4f}")

    except Exception as e:
        print(f"❌ Erreur sur dataset {dataset_letter}: {e}")
```

## Détection automatique

Le système détecte automatiquement :

1. **Format des données** :
   - Si le répertoire contient `.data` + `.solution` → Format Challenge
   - Sinon → Format fichier unique

2. **Séparateur** (pour fichiers .data) :
   - Espaces (par défaut pour Challenge)
   - Tabulations, virgules, points-virgules (essayés si espaces échouent)

3. **Type de tâche** :
   - Classification : < 20 valeurs uniques ET < 5% du total
   - Régression : sinon

## Tester votre configuration

Utilisez le script de test fourni :

```bash
python test_challenge_format.py
```

Modifiez les variables en haut du script :
- `DATASET = 'D'` → Lettre de votre dataset
- `BASE_PATH = "/info/corpus/ChallengeMachineLearning"` → Chemin de base

## Récapitulatif des changements

Le DataLoader a été modifié pour :

✅ Détecter automatiquement le format Challenge (`.data` + `.solution`)
✅ Charger les features depuis `data_X.data` (séparées par espaces)
✅ Charger le target depuis `data_X.solution`
✅ Rester compatible avec l'ancien format (fichier unique)
✅ Ne pas modifier le reste du pipeline (preprocessing, models, evaluation)

## Exemple de structure de fichiers

### data_A.data (extrait)
```
1.0 2.5 3.2 4.1 5.0
2.0 3.1 4.3 5.2 6.1
3.0 1.8 2.9 3.5 4.2
```

### data_A.solution (extrait)
```
0
1
0
```

### Utilisation
```python
import automl

# Pointer vers le répertoire (pas le fichier)
automl.fit(data_path="/info/corpus/ChallengeMachineLearning/data_A")
```

Le système :
1. Détecte qu'il y a `data_A.data` et `data_A.solution`
2. Charge X depuis `.data` (séparé par espaces)
3. Charge y depuis `.solution`
4. Continue normalement avec preprocessing → training → evaluation

## Comparaison avec votre code original

Votre code :
```python
X = pd.read_csv(data_path / f"data_{dataset_name}.data", sep=' ', header=None)
y = pd.read_csv(data_path / f"data_{dataset_name}.solution", header=None, names=['target'])['target']
```

AutoML maintenant :
```python
automl.fit(data_path=f"{base_path}/data_{dataset_name}")
# Fait exactement la même chose + preprocessing + training + evaluation
```

C'est équivalent, mais AutoML ajoute :
- Prétraitement automatique
- Entraînement de 6-7 modèles
- Sélection du meilleur
- Évaluation complète
- Métriques détaillées
