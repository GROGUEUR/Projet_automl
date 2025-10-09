import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import os

class AutoML:
    """
    Classe principale pour l'AutoML
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def fit(self, data_path):
        """
        Entraîne automatiquement des modèles sur les données
        
        Args:
            data_path (str): Chemin vers le répertoire contenant les données
        """
        print(f"📂 Chargement des données depuis {data_path}")
        
        # 1. Charger les données
        self._load_data(data_path)
        
        # 2. Préparer les données
        self._prepare_data()
        
        # 3. Entraîner plusieurs modèles
        self._train_models()
        
        # 4. Sélectionner le meilleur
        self._select_best_model()
        
        print(f"✅ Entraînement terminé ! Meilleur modèle : {self.best_model['name']}")
        
    def _load_data(self, data_path):
        """Charge les données depuis le chemin spécifié"""
        # TODO: Implémenter le chargement
        # Lister les fichiers CSV dans data_path
        # Charger le premier fichier pour commencer
        files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        if files:
            df = pd.read_csv(os.path.join(data_path, files[0]))
            print(f"   Données chargées : {df.shape}")
            self.data = df
        else:
            raise ValueError("Aucun fichier CSV trouvé")
    
    def _prepare_data(self):
        """Prépare les données (train/test split, etc.)"""
        # TODO: Implémenter la préparation
        # Supposons que la dernière colonne est la cible
        X = self.data.iloc[:, :-1]
        y = self.data.iloc[:, -1]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"   Train set : {self.X_train.shape}")
        print(f"   Test set : {self.X_test.shape}")
    
    def _train_models(self):
        """Entraîne plusieurs modèles"""
        print("🤖 Entraînement des modèles...")
        
        # Liste des modèles à tester
        models_to_test = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        for name, model in models_to_test.items():
            print(f"   Entraînement de {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Évaluer sur le set de validation
            score = model.score(self.X_test, self.y_test)
            
            self.models[name] = {
                'model': model,
                'score': score,
                'name': name
            }
            print(f"      Score : {score:.4f}")
    
    def _select_best_model(self):
        """Sélectionne le meilleur modèle"""
        best_name = max(self.models, key=lambda k: self.models[k]['score'])
        self.best_model = self.models[best_name]
    
    def eval(self):
        """Évalue le meilleur modèle"""
        if self.best_model is None:
            raise ValueError("Aucun modèle entraîné. Appelez fit() d'abord.")
        
        print(f"\n📊 Évaluation du meilleur modèle : {self.best_model['name']}")
        print(f"   Score : {self.best_model['score']:.4f}")
        
        # Prédictions
        y_pred = self.best_model['model'].predict(self.X_test)
        
        # Rapport détaillé
        print("\n" + classification_report(self.y_test, y_pred))
