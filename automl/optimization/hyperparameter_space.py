class HyperparameterSpace:
    """
    Contient les espaces de recherche d'hyperparamètres optimisés.
    """
    
    # Random Forest
    RANDOM_FOREST_SPACE = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'] 
    }
    
    # Gradient Boosting
    GRADIENT_BOOSTING_SPACE = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],         
        'min_samples_split': [2, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Logistic Regression
    LOGISTIC_SPACE = {
        'C': [0.01, 0.1, 1, 10],       
        'solver': ['lbfgs']            
    }

    # Ridge
    RIDGE_SPACE = {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
    
    # SVM 
    SVM_SPACE = {
        'C': [0.1, 1, 10],             
        'kernel': ['linear', 'rbf'],   
        'gamma': ['scale', 'auto']
    }
    
    # KNN
    KNN_SPACE = {
        'n_neighbors': [3, 5, 7, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # Decision Tree
    DECISION_TREE_SPACE = {
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Naive Bayes
    NAIVE_BAYES_SPACE = {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
    
    @staticmethod
    def get_space(model_name: str, task_type: str = None):
        """
        Retourne l'espace de recherche pour un modèle donné.
        """
        # Mise à jour du mapping pour utiliser les nouveaux dictionnaires
        spaces = {
            'RandomForest': HyperparameterSpace.RANDOM_FOREST_SPACE,
            'GradientBoosting': HyperparameterSpace.GRADIENT_BOOSTING_SPACE,
            'LogisticRegression': HyperparameterSpace.LOGISTIC_SPACE, 
            'Ridge': HyperparameterSpace.RIDGE_SPACE,                 
            'SVM': HyperparameterSpace.SVM_SPACE,
            'SVR': HyperparameterSpace.SVM_SPACE,
            'KNN': HyperparameterSpace.KNN_SPACE,
            'DecisionTree': HyperparameterSpace.DECISION_TREE_SPACE,
            'NaiveBayes': HyperparameterSpace.NAIVE_BAYES_SPACE
        }
        
        space = spaces.get(model_name, {})
        
        # Adapter selon le type de tâche
        if task_type == 'regression' and model_name == 'DecisionTree':
            space = space.copy()
            space['criterion'] = ['squared_error', 'absolute_error']
        
        # Gestion spécifique pour LogisticRegression solver
        if model_name == 'LogisticRegression':
             space = space.copy()
        
        return space
    
    @staticmethod
    def get_reduced_space(model_name: str, task_type: str = None):
        """
        Retourne un espace réduit pour recherche rapide.
        """
        full_space = HyperparameterSpace.get_space(model_name, task_type)
        reduced_space = {}
        
        for key, values in full_space.items():
            if isinstance(values, list):
                # On prend max 2 valeurs pour aller très vite en mode debug/réduit
                n = len(values)
                if n <= 2:
                    reduced_space[key] = values
                else:
                    # On prend juste le premier et le dernier (extrêmes)
                    reduced_space[key] = [values[0], values[-1]]
            else:
                reduced_space[key] = values
        
        return reduced_space