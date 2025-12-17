"""
Tests unitaires pour le module data/loader.py
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from pathlib import Path

from automl.data.loader import DataLoader
from automl.data.preprocessing import DataPreprocessor, train_valid_test_split


class TestDataLoader:
    """Tests pour la classe DataLoader."""

    @pytest.fixture
    def temp_csv_classification(self):
        """Crée un fichier CSV temporaire pour la classification."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,5.0,6.0,1\n")
            f.write("7.0,8.0,9.0,0\n")
            f.write("10.0,11.0,12.0,1\n")
            f.write("13.0,14.0,15.0,0\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_regression(self):
        """Crée un fichier CSV temporaire pour la régression."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1.0,2.0,3.0,10.5\n")
            f.write("4.0,5.0,6.0,20.3\n")
            f.write("7.0,8.0,9.0,30.7\n")
            f.write("10.0,11.0,12.0,40.2\n")
            f.write("13.0,14.0,15.0,50.9\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_with_missing(self):
        """Crée un fichier CSV avec valeurs manquantes."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("1.0,2.0,3.0,0\n")
            f.write("4.0,,6.0,1\n")
            f.write("7.0,8.0,,0\n")
            f.write(",11.0,12.0,1\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_txt_space_separated(self):
        """Crée un fichier TXT avec séparateur espace."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("1.0 2.0 3.0 0\n")
            f.write("4.0 5.0 6.0 1\n")
            f.write("7.0 8.0 9.0 0\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    @pytest.fixture
    def temp_csv_categorical(self):
        """Crée un fichier CSV avec variables catégorielles."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("red,big,1.0,yes\n")
            f.write("blue,small,2.0,no\n")
            f.write("red,medium,3.0,yes\n")
            f.write("green,big,4.0,no\n")
            temp_path = f.name
        yield temp_path
        os.unlink(temp_path)

    def test_load_csv_classification(self, temp_csv_classification):
        """Test du chargement d'un CSV pour classification."""
        loader = DataLoader(temp_csv_classification)
        X, y, task_type = loader.load_data()

        assert X.shape == (5, 3)
        assert y.shape == (5,)
        assert task_type == 'classification'
        assert len(np.unique(y)) == 2

    def test_load_csv_regression(self, temp_csv_regression):
        """Test du chargement d'un CSV pour régression."""
        loader = DataLoader(temp_csv_regression)
        X, y, task_type = loader.load_data()

        assert X.shape == (5, 3)
        assert y.shape == (5,)
        assert task_type == 'regression'

    def test_load_txt_file(self, temp_txt_space_separated):
        """Test du chargement d'un fichier TXT."""
        loader = DataLoader(temp_txt_space_separated)
        X, y, task_type = loader.load_data()

        assert X.shape == (3, 3)
        assert y.shape == (3,)

    def test_detect_task_type_classification(self):
        """Test de la détection du type classification."""
        loader = DataLoader.__new__(DataLoader)
        y_class = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        task_type = loader.detect_task_type(y_class)
        assert task_type == 'classification'

    def test_detect_task_type_regression(self):
        """Test de la détection du type régression."""
        loader = DataLoader.__new__(DataLoader)
        y_reg = np.array([1.5, 2.7, 3.2, 4.9, 5.1, 6.3, 7.8, 8.2])
        task_type = loader.detect_task_type(y_reg)
        assert task_type == 'regression'

    def test_file_not_found(self):
        """Test avec un fichier inexistant."""
        with pytest.raises(FileNotFoundError):
            DataLoader("/path/that/does/not/exist.csv")

    def test_missing_values_detection(self, temp_csv_with_missing):
        """Test de la détection des valeurs manquantes."""
        loader = DataLoader(temp_csv_with_missing)
        X, y, task_type = loader.load_data()

        # Vérifier qu'il y a bien des valeurs manquantes
        assert np.sum(pd.isna(X)) > 0

    def test_get_info(self, temp_csv_classification):
        """Test de la méthode get_info."""
        loader = DataLoader(temp_csv_classification)
        loader.load_data()
        info = loader.get_info()

        assert 'n_samples' in info
        assert 'n_features' in info
        assert 'task_type' in info
        assert info['n_samples'] == 5
        assert info['n_features'] == 3

    def test_categorical_features(self, temp_csv_categorical):
        """Test avec des features catégorielles."""
        loader = DataLoader(temp_csv_categorical)
        X, y, task_type = loader.load_data()

        assert X.shape[0] == 4
        assert X.shape[1] == 3


class TestDataPreprocessor:
    """Tests pour la classe DataPreprocessor."""

    @pytest.fixture
    def sample_numeric_data(self):
        """Données numériques pour les tests."""
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ])
        y = np.array([0, 1, 0, 1])
        return X, y

    @pytest.fixture
    def sample_data_with_missing(self):
        """Données avec valeurs manquantes."""
        X = np.array([
            [1.0, 2.0, 3.0],
            [4.0, np.nan, 6.0],
            [7.0, 8.0, np.nan],
            [np.nan, 11.0, 12.0]
        ])
        y = np.array([0, 1, 0, 1])
        return X, y

    @pytest.fixture
    def sample_mixed_data(self):
        """Données mixtes numériques et catégorielles."""
        X = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0],
            'num2': [5.0, 6.0, 7.0, 8.0],
            'cat1': ['A', 'B', 'A', 'C'],
            'cat2': ['X', 'Y', 'X', 'Y']
        })
        y = np.array([0, 1, 0, 1])
        return X, y

    def test_fit_transform_numeric(self, sample_numeric_data):
        """Test du fit_transform sur données numériques."""
        X, y = sample_numeric_data
        preprocessor = DataPreprocessor(scale=True)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape == X.shape
        assert preprocessor.is_fitted
        # Vérifier la normalisation (moyenne ~0, std ~1)
        assert np.abs(np.mean(X_transformed)) < 1e-10
        assert np.abs(np.std(X_transformed) - 1.0) < 1e-10

    def test_handle_missing_mean(self, sample_data_with_missing):
        """Test de la gestion des valeurs manquantes avec mean."""
        X, y = sample_data_with_missing
        preprocessor = DataPreprocessor(handle_missing='mean', scale=False)
        X_transformed = preprocessor.fit_transform(X)

        # Vérifier qu'il n'y a plus de valeurs manquantes
        assert not np.any(np.isnan(X_transformed))

    def test_handle_missing_median(self, sample_data_with_missing):
        """Test de la gestion des valeurs manquantes avec median."""
        X, y = sample_data_with_missing
        preprocessor = DataPreprocessor(handle_missing='median', scale=False)
        X_transformed = preprocessor.fit_transform(X)

        assert not np.any(np.isnan(X_transformed))

    def test_categorical_encoding(self, sample_mixed_data):
        """Test de l'encodage des variables catégorielles."""
        X, y = sample_mixed_data
        preprocessor = DataPreprocessor(
            encode_categorical=True,
            scale=False,
            handle_missing='most_frequent'
        )
        X_transformed = preprocessor.fit_transform(X)

        # Vérifier que les données sont bien encodées (tout numérique)
        assert X_transformed.dtype in [np.float64, np.int64]
        assert X_transformed.shape[0] == X.shape[0]

    def test_transform_without_fit(self, sample_numeric_data):
        """Test que transform échoue si fit n'a pas été appelé."""
        X, y = sample_numeric_data
        preprocessor = DataPreprocessor()

        with pytest.raises(RuntimeError):
            preprocessor.transform(X)

    def test_fit_and_transform_separately(self, sample_numeric_data):
        """Test de fit et transform séparément."""
        X, y = sample_numeric_data
        preprocessor = DataPreprocessor(scale=True)

        preprocessor.fit(X)
        X_transformed = preprocessor.transform(X)

        assert X_transformed.shape == X.shape
        assert preprocessor.is_fitted

    def test_save_and_load(self, sample_numeric_data, tmp_path):
        """Test de la sauvegarde et du chargement du preprocessor."""
        X, y = sample_numeric_data
        preprocessor = DataPreprocessor(scale=True)
        preprocessor.fit(X)

        # Sauvegarder
        save_path = preprocessor.save(str(tmp_path))
        assert os.path.exists(save_path)

        # Charger
        loaded_preprocessor = DataPreprocessor.load(save_path)
        assert loaded_preprocessor.is_fitted

        # Vérifier que la transformation est identique
        X_transformed1 = preprocessor.transform(X)
        X_transformed2 = loaded_preprocessor.transform(X)
        np.testing.assert_array_almost_equal(X_transformed1, X_transformed2)

    def test_invalid_strategy(self):
        """Test avec une stratégie invalide."""
        with pytest.raises(ValueError):
            DataPreprocessor(handle_missing='invalid_strategy')


class TestTrainValidTestSplit:
    """Tests pour la fonction train_valid_test_split."""

    @pytest.fixture
    def sample_data(self):
        """Données pour les tests de split."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        return X, y

    def test_split_proportions(self, sample_data):
        """Test que les proportions sont respectées."""
        X, y = sample_data
        splits = train_valid_test_split(
            X, y,
            train_size=0.7,
            valid_size=0.15,
            test_size=0.15
        )

        n_total = len(X)
        n_train = len(splits['X_train'])
        n_valid = len(splits['X_valid'])
        n_test = len(splits['X_test'])

        # Vérifier les proportions (avec une petite tolérance)
        assert abs(n_train / n_total - 0.7) < 0.05
        assert abs(n_valid / n_total - 0.15) < 0.05
        assert abs(n_test / n_total - 0.15) < 0.05

        # Vérifier que la somme fait 100%
        assert n_train + n_valid + n_test == n_total

    def test_split_stratification(self, sample_data):
        """Test de la stratification pour la classification."""
        X, y = sample_data
        splits = train_valid_test_split(
            X, y,
            task_type='classification'
        )

        # Vérifier que les proportions de classes sont similaires
        train_ratio = np.mean(splits['y_train'])
        valid_ratio = np.mean(splits['y_valid'])
        test_ratio = np.mean(splits['y_test'])
        overall_ratio = np.mean(y)

        # Les ratios devraient être proches
        assert abs(train_ratio - overall_ratio) < 0.15
        assert abs(valid_ratio - overall_ratio) < 0.15
        assert abs(test_ratio - overall_ratio) < 0.15

    def test_split_invalid_proportions(self, sample_data):
        """Test avec des proportions invalides."""
        X, y = sample_data

        with pytest.raises(ValueError):
            train_valid_test_split(
                X, y,
                train_size=0.5,
                valid_size=0.3,
                test_size=0.3  # Total = 1.1 > 1.0
            )

    def test_split_reproducibility(self, sample_data):
        """Test que le split est reproductible avec le même random_state."""
        X, y = sample_data

        splits1 = train_valid_test_split(X, y, random_state=42)
        splits2 = train_valid_test_split(X, y, random_state=42)

        np.testing.assert_array_equal(splits1['X_train'], splits2['X_train'])
        np.testing.assert_array_equal(splits1['y_train'], splits2['y_train'])

    def test_split_returns_dict(self, sample_data):
        """Test que le split retourne bien un dictionnaire avec les bonnes clés."""
        X, y = sample_data
        splits = train_valid_test_split(X, y)

        expected_keys = ['X_train', 'X_valid', 'X_test', 'y_train', 'y_valid', 'y_test']
        assert all(key in splits for key in expected_keys)

    def test_split_with_pandas(self, sample_data):
        """Test du split avec des DataFrames pandas."""
        X, y = sample_data
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)

        splits = train_valid_test_split(X_df, y_series)

        # Vérifier que ça fonctionne aussi avec pandas
        assert splits['X_train'] is not None
        assert len(splits['X_train']) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
