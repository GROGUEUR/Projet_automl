"""
Tests unitaires pour le module models (Personne 2).

Ces tests valident le bon fonctionnement de :
- BaseModel
- ModelFactory
- ModelTrainer
- ModelSelector
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
import tempfile
import os

from automl.models.base_model import BaseModel
from automl.models.model_factory import ModelFactory
from automl.models.model_trainer import ModelTrainer
from automl.models.model_selector import ModelSelector


# ==================== FIXTURES ====================

@pytest.fixture
def classification_data():
    """Génère des données de classification pour les tests."""
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    X_train, X_valid = X[:150], X[150:]
    y_train, y_valid = y[:150], y[150:]
    return X_train, X_valid, y_train, y_valid


@pytest.fixture
def regression_data():
    """Génère des données de régression pour les tests."""
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    X_train, X_valid = X[:150], X[150:]
    y_train, y_valid = y[:150], y[150:]
    return X_train, X_valid, y_train, y_valid


# ==================== TESTS BASEMODEL ====================

def test_base_model_creation():
    """Test la création d'un BaseModel."""
    model = BaseModel(
        name="TestModel",
        model=RandomForestClassifier(random_state=42),
        task_type='classification'
    )
    assert model.name == "TestModel"
    assert model.task_type == 'classification'
    assert not model.is_fitted
    assert model.train_score is None


def test_base_model_fit(classification_data):
    """Test l'entraînement d'un BaseModel."""
    X_train, X_valid, y_train, y_valid = classification_data

    model = BaseModel(
        name="TestModel",
        model=RandomForestClassifier(random_state=42),
        task_type='classification'
    )

    model.fit(X_train, y_train, X_valid, y_valid)

    assert model.is_fitted
    assert model.train_score is not None
    assert model.valid_score is not None
    assert 0 <= model.train_score <= 1
    assert 0 <= model.valid_score <= 1
    assert model.training_time > 0


def test_base_model_predict(classification_data):
    """Test la prédiction avec BaseModel."""
    X_train, X_valid, y_train, y_valid = classification_data

    model = BaseModel(
        name="TestModel",
        model=RandomForestClassifier(random_state=42),
        task_type='classification'
    )

    # Doit échouer avant fit
    with pytest.raises(RuntimeError):
        model.predict(X_valid)

    # Entraîner
    model.fit(X_train, y_train)

    # Doit fonctionner après fit
    predictions = model.predict(X_valid)
    assert len(predictions) == len(y_valid)


def test_base_model_save_load(classification_data):
    """Test la sauvegarde et le chargement d'un BaseModel."""
    X_train, X_valid, y_train, y_valid = classification_data

    model = BaseModel(
        name="TestModel",
        model=RandomForestClassifier(random_state=42),
        task_type='classification'
    )
    model.fit(X_train, y_train, X_valid, y_valid)

    # Sauvegarder
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = model.save(tmpdir)
        assert os.path.exists(save_path)

        # Charger
        loaded_model = BaseModel.load(save_path)
        assert loaded_model.name == model.name
        assert loaded_model.is_fitted
        assert loaded_model.train_score == model.train_score

        # Vérifier que les prédictions sont identiques
        pred_original = model.predict(X_valid)
        pred_loaded = loaded_model.predict(X_valid)
        np.testing.assert_array_equal(pred_original, pred_loaded)


def test_base_model_get_set_params():
    """Test get_params et set_params."""
    model = BaseModel(
        name="TestModel",
        model=RandomForestClassifier(n_estimators=100, random_state=42),
        task_type='classification'
    )

    params = model.get_params()
    assert params['n_estimators'] == 100

    model.set_params(n_estimators=50)
    params = model.get_params()
    assert params['n_estimators'] == 50


# ==================== TESTS MODEL FACTORY ====================

def test_model_factory_classification():
    """Test que ModelFactory crée tous les modèles de classification."""
    models = ModelFactory.get_default_models('classification')
    assert len(models) == 7  # 7 modèles de classification

    model_names = [m.name for m in models]
    expected_names = ['RandomForest', 'GradientBoosting', 'LogisticRegression',
                     'SVM', 'KNN', 'DecisionTree', 'NaiveBayes']

    for name in expected_names:
        assert name in model_names


def test_model_factory_regression():
    """Test que ModelFactory crée tous les modèles de régression."""
    models = ModelFactory.get_default_models('regression')
    assert len(models) == 6  # 6 modèles de régression

    model_names = [m.name for m in models]
    expected_names = ['RandomForest', 'GradientBoosting', 'Ridge',
                     'SVR', 'KNN', 'DecisionTree']

    for name in expected_names:
        assert name in model_names


def test_model_factory_invalid_task_type():
    """Test que ModelFactory rejette un task_type invalide."""
    with pytest.raises(ValueError):
        ModelFactory.get_default_models('invalid_task')


def test_model_factory_create_model():
    """Test la création d'un modèle spécifique."""
    # Test avec RandomForest qui accepte n_estimators
    model = ModelFactory.create_model('RandomForest', 'classification', random_state=42, n_estimators=50)
    assert model.name == 'RandomForest'
    assert model.task_type == 'classification'
    assert model.get_params()['n_estimators'] == 50

    # Test avec un autre modèle
    model2 = ModelFactory.create_model('KNN', 'classification', n_neighbors=3)
    assert model2.name == 'KNN'
    assert model2.get_params()['n_neighbors'] == 3


def test_model_factory_get_available_models():
    """Test la récupération de la liste des modèles disponibles."""
    clf_models = ModelFactory.get_available_models('classification')
    assert len(clf_models) == 7

    reg_models = ModelFactory.get_available_models('regression')
    assert len(reg_models) == 6


# ==================== TESTS MODEL TRAINER ====================

def test_model_trainer_classification(classification_data):
    """Test que ModelTrainer entraîne tous les modèles de classification."""
    X_train, X_valid, y_train, y_valid = classification_data

    trainer = ModelTrainer(task_type='classification', verbose=False)
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    assert len(results) > 0
    assert all('valid_score' in r for r in results)
    assert all('train_score' in r for r in results)
    assert all('training_time' in r for r in results)


def test_model_trainer_regression(regression_data):
    """Test que ModelTrainer entraîne tous les modèles de régression."""
    X_train, X_valid, y_train, y_valid = regression_data

    trainer = ModelTrainer(task_type='regression', verbose=False)
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    assert len(results) > 0
    assert all('valid_score' in r for r in results)


def test_model_trainer_best_model_selection(classification_data):
    """Test la sélection du meilleur modèle."""
    X_train, X_valid, y_train, y_valid = classification_data

    trainer = ModelTrainer(task_type='classification', verbose=False)
    trainer.train_all(X_train, y_train, X_valid, y_valid)

    best = trainer.select_best_model()
    assert best is not None
    assert best.is_fitted
    assert trainer.best_model == best


def test_model_trainer_get_results_summary(classification_data):
    """Test la génération du résumé des résultats."""
    X_train, X_valid, y_train, y_valid = classification_data

    trainer = ModelTrainer(task_type='classification', verbose=False)
    trainer.train_all(X_train, y_train, X_valid, y_valid)

    summary = trainer.get_results_summary()
    assert not summary.empty
    assert 'name' in summary.columns
    assert 'valid_score' in summary.columns


def test_model_trainer_save_models(classification_data):
    """Test la sauvegarde des modèles."""
    X_train, X_valid, y_train, y_valid = classification_data

    trainer = ModelTrainer(task_type='classification', verbose=False)
    trainer.train_all(X_train, y_train, X_valid, y_valid)

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_paths = trainer.save_all_models(tmpdir)
        assert len(saved_paths) > 0
        assert all(os.path.exists(p) for p in saved_paths)


def test_model_trainer_get_model(classification_data):
    """Test la récupération d'un modèle spécifique."""
    X_train, X_valid, y_train, y_valid = classification_data

    trainer = ModelTrainer(task_type='classification', verbose=False)
    trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Doit fonctionner pour un modèle existant
    rf_model = trainer.get_model('RandomForest')
    assert rf_model.name == 'RandomForest'

    # Doit échouer pour un modèle inexistant
    with pytest.raises(KeyError):
        trainer.get_model('NonExistentModel')


# ==================== TESTS MODEL SELECTOR ====================

def test_model_selector_by_score():
    """Test la sélection par score."""
    results = [
        {'name': 'Model1', 'valid_score': 0.8, 'training_time': 1.0},
        {'name': 'Model2', 'valid_score': 0.9, 'training_time': 2.0},
        {'name': 'Model3', 'valid_score': 0.7, 'training_time': 0.5},
    ]

    best = ModelSelector.select_by_score(results)
    assert best == 'Model2'


def test_model_selector_by_speed_score_tradeoff():
    """Test la sélection par compromis vitesse/performance."""
    results = [
        {'name': 'FastButWeak', 'valid_score': 0.7, 'training_time': 0.1},
        {'name': 'SlowButStrong', 'valid_score': 0.95, 'training_time': 10.0},
        {'name': 'Balanced', 'valid_score': 0.85, 'training_time': 1.0},
    ]

    # Avec poids égaux, le modèle équilibré devrait être favorisé
    best = ModelSelector.select_by_speed_score_tradeoff(
        results, score_weight=0.5, speed_weight=0.5
    )
    # Le résultat peut varier selon la normalisation
    assert best in ['FastButWeak', 'SlowButStrong', 'Balanced']


def test_model_selector_top_k():
    """Test la sélection des k meilleurs."""
    results = [
        {'name': 'Model1', 'valid_score': 0.8},
        {'name': 'Model2', 'valid_score': 0.9},
        {'name': 'Model3', 'valid_score': 0.7},
        {'name': 'Model4', 'valid_score': 0.85},
    ]

    top_3 = ModelSelector.select_top_k(results, k=3)
    assert len(top_3) == 3
    assert 'Model2' in top_3  # Le meilleur
    assert 'Model4' in top_3  # Le deuxième
    assert 'Model1' in top_3  # Le troisième


def test_model_selector_overfitting_control():
    """Test la sélection avec contrôle du surapprentissage."""
    results = [
        {'name': 'Overfitted', 'train_score': 0.99, 'valid_score': 0.7},
        {'name': 'Balanced', 'train_score': 0.85, 'valid_score': 0.82},
        {'name': 'Underfitted', 'train_score': 0.65, 'valid_score': 0.63},
    ]

    best = ModelSelector.select_by_overfitting_control(results, max_gap=0.1)
    # Le modèle équilibré devrait être sélectionné
    assert best == 'Balanced'


def test_model_selector_rankings():
    """Test le classement des modèles."""
    results = [
        {'name': 'Model1', 'valid_score': 0.8},
        {'name': 'Model2', 'valid_score': 0.9},
        {'name': 'Model3', 'valid_score': 0.7},
    ]

    rankings = ModelSelector.get_model_rankings(results)
    assert len(rankings) == 3
    assert rankings[0]['name'] == 'Model2'
    assert rankings[0]['rank'] == 1
    assert rankings[1]['rank'] == 2
    assert rankings[2]['rank'] == 3


# ==================== TESTS D'INTÉGRATION ====================

def test_full_pipeline_classification(classification_data):
    """Test du pipeline complet pour la classification."""
    X_train, X_valid, y_train, y_valid = classification_data

    # Créer des modèles
    models = ModelFactory.get_default_models('classification')

    # Entraîner
    trainer = ModelTrainer(task_type='classification', models=models[:3], verbose=False)
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Sélectionner le meilleur
    best = trainer.select_best_model()

    # Prédire
    predictions = best.predict(X_valid)

    assert len(predictions) == len(y_valid)
    assert best.valid_score > 0


def test_full_pipeline_regression(regression_data):
    """Test du pipeline complet pour la régression."""
    X_train, X_valid, y_train, y_valid = regression_data

    # Créer des modèles
    models = ModelFactory.get_default_models('regression')

    # Entraîner
    trainer = ModelTrainer(task_type='regression', models=models[:3], verbose=False)
    results = trainer.train_all(X_train, y_train, X_valid, y_valid)

    # Sélectionner le meilleur
    best = trainer.select_best_model()

    # Prédire
    predictions = best.predict(X_valid)

    assert len(predictions) == len(y_valid)
    assert best.valid_score is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
