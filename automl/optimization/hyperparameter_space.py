"""
Définition des espaces de recherche d'hyperparamètres pour chaque modèle.
"""

class HyperparameterSpace:
    """
    Contient les espaces de recherche d'hyperparamètres pour tous les modèles.
    """
    
    # Random Forest
    RANDOM_FOREST_SPACE = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Gradient Boosting
    GRADIENT_BOOSTING_SPACE = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Logistic Regression / Ridge
    LINEAR_SPACE = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Pour Logistic
        'alpha': [0.001, 0.01, 0.1, 1, 10, 100],  # Pour Ridge
        'solver': ['lbfgs', 'saga']  # Pour Logistic
    }
    
    # SVM / SVR
    SVM_SPACE = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
    }
    
    # KNN
    KNN_SPACE = {
        'n_neighbors': [3, 5, 7, 10, 15, 20],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Decision Tree
    DECISION_TREE_SPACE = {
        'max_depth': [None, 5, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'criterion': ['gini', 'entropy']  # Pour classification
    }
    
    # Naive Bayes
    NAIVE_BAYES_SPACE = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
    }
    
    @staticmethod
    def get_space(model_name: str, task_type: str = None):
        """
        Retourne l'espace de recherche pour un modèle donné.
        """
        spaces = {
            'RandomForest': HyperparameterSpace.RANDOM_FOREST_SPACE,
            'GradientBoosting': HyperparameterSpace.GRADIENT_BOOSTING_SPACE,
            'LogisticRegression': HyperparameterSpace.LINEAR_SPACE,
            'Ridge': HyperparameterSpace.LINEAR_SPACE,
            'SVM': HyperparameterSpace.SVM_SPACE,
            'SVR': HyperparameterSpace.SVM_SPACE,
            'KNN': HyperparameterSpace.KNN_SPACE,
            'DecisionTree': HyperparameterSpace.DECISION_TREE_SPACE,
            'NaiveBayes': HyperparameterSpace.NAIVE_BAYES_SPACE
        }
        
        space = spaces.get(model_name, {})
        
        # Adapter selon le type de tâche si nécessaire
        if task_type == 'regression' and model_name == 'DecisionTree':
            space = space.copy()
            space['criterion'] = ['squared_error', 'absolute_error']
        
        return space
    
    @staticmethod
    def get_reduced_space(model_name: str, task_type: str = None):
        """
        Retourne un espace réduit pour recherche rapide (3-5 valeurs par param).
        """
        full_space = HyperparameterSpace.get_space(model_name, task_type)
        
        # Réduire chaque liste à max 3 valeurs (début, milieu, fin)
        reduced_space = {}
        for key, values in full_space.items():
            if isinstance(values, list):
                n = len(values)
                if n <= 3:
                    reduced_space[key] = values
                else:
                    indices = [0, n//2, n-1]
                    reduced_space[key] = [values[i] for i in indices]
            else:
                reduced_space[key] = values
        
        return reduced_space