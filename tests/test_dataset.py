"""
Tests para el módulo de dataset.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_project.dataset import DataCleaner, DataLoader, DataSplitter


class TestDataLoader(unittest.TestCase):
    """Tests para la clase DataLoader."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear un archivo CSV temporal para testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name) / "test_data.csv"

        # Crear datos de prueba
        test_data = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )
        test_data.to_csv(self.temp_path, index=False)

    def tearDown(self):
        """Limpieza después de cada test."""
        self.temp_dir.cleanup()

    def test_loader_initialization(self):
        """Verifica que el loader se inicialice correctamente."""
        loader = DataLoader(self.temp_path)
        self.assertEqual(loader.filepath, self.temp_path)
        self.assertEqual(loader.delimiter, ",")
        self.assertIsNone(loader.data)

    def test_load_data_success(self):
        """Verifica que se carguen datos correctamente."""
        loader = DataLoader(self.temp_path)
        data = loader.load_data()

        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 5)
        self.assertEqual(len(data.columns), 3)

    def test_load_data_file_not_found(self):
        """Verifica el comportamiento cuando el archivo no existe."""
        loader = DataLoader("nonexistent_file.csv")
        data = loader.load_data()

        self.assertIsNone(data)

    def test_get_data_before_loading(self):
        """Verifica get_data antes de cargar."""
        loader = DataLoader(self.temp_path)
        data = loader.get_data()

        self.assertIsNone(data)

    def test_get_data_after_loading(self):
        """Verifica get_data después de cargar."""
        loader = DataLoader(self.temp_path)
        loader.load_data()
        data = loader.get_data()

        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)


class TestDataCleaner(unittest.TestCase):
    """Tests para la clase DataCleaner."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear datos de prueba con valores problemáticos
        self.test_data = pd.DataFrame(
            {
                "feature1": [1, 2, "invalid", 4, 5],
                "feature2": [10, 20, 30, "error", 50],
                "kredit": [0, 1, 0, np.nan, 1],
                "mixed_type_col": ["a", "b", "c", "d", "e"],
            }
        )

    def test_cleaner_initialization(self):
        """Verifica que el cleaner se inicialice correctamente."""
        cleaner = DataCleaner()
        self.assertEqual(cleaner.target_column, "kredit")
        self.assertEqual(cleaner.mixed_type_column, "mixed_type_col")

    def test_clean_data_removes_mixed_column(self):
        """Verifica que se elimine la columna de tipo mixto."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(self.test_data)

        self.assertNotIn("mixed_type_col", cleaned.columns)

    def test_clean_data_converts_to_numeric(self):
        """Verifica que se conviertan valores a numérico."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(self.test_data)

        # Verificar que las columnas sean numéricas
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["feature1"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(cleaned["feature2"]))

    def test_clean_data_removes_invalid_target(self):
        """Verifica que se eliminen filas con target inválido."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(self.test_data)

        # La fila con target NaN debe ser eliminada
        self.assertEqual(len(cleaned), 4)  # 5 originales - 1 con NaN
        self.assertFalse(cleaned["kredit"].isna().any())

    def test_clean_data_preserves_valid_data(self):
        """Verifica que se preserven los datos válidos."""
        # Datos completamente válidos
        valid_data = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [10, 20, 30], "kredit": [0, 1, 0]}
        )

        cleaner = DataCleaner(mixed_type_column=None)
        cleaned = cleaner.clean_data(valid_data)

        self.assertEqual(len(cleaned), len(valid_data))


class TestDataSplitter(unittest.TestCase):
    """Tests para la clase DataSplitter."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear datos de prueba
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
                "target": np.random.choice([0, 1], 100),
            }
        )

    def test_splitter_initialization(self):
        """Verifica que el splitter se inicialice correctamente."""
        splitter = DataSplitter(test_size=0.2, random_state=42)

        self.assertEqual(splitter.test_size, 0.2)
        self.assertEqual(splitter.random_state, 42)
        self.assertTrue(splitter.stratify)

    def test_split_returns_correct_shapes(self):
        """Verifica que split retorne las formas correctas."""
        splitter = DataSplitter(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split(self.test_data, "target")

        # Verificar que se dividió correctamente
        total_samples = len(self.test_data)
        expected_test_size = int(total_samples * 0.3)

        self.assertEqual(len(X_train) + len(X_test), total_samples)
        self.assertEqual(len(y_train), len(X_train))
        self.assertEqual(len(y_test), len(X_test))
        self.assertAlmostEqual(len(X_test), expected_test_size, delta=2)

    def test_split_separates_features_and_target(self):
        """Verifica que se separen features y target correctamente."""
        splitter = DataSplitter(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splitter.split(self.test_data, "target")

        # Verificar que X no contenga el target
        self.assertNotIn("target", X_train.columns)
        self.assertNotIn("target", X_test.columns)

        # Verificar que y sea una Serie
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)

    def test_split_invalid_target_column(self):
        """Verifica el comportamiento con columna objetivo inválida."""
        splitter = DataSplitter()

        with self.assertRaises(ValueError):
            splitter.split(self.test_data, "nonexistent_column")

    def test_split_stratification(self):
        """Verifica que la estratificación funcione correctamente."""
        # Crear datos con clases claramente desbalanceadas
        unbalanced_data = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "target": [0] * 70 + [1] * 30,  # 70% clase 0, 30% clase 1
            }
        )

        splitter = DataSplitter(test_size=0.3, random_state=42, stratify=True)
        X_train, X_test, y_train, y_test = splitter.split(unbalanced_data, "target")

        # Verificar que las proporciones se mantengan aproximadamente
        train_ratio = y_train.value_counts(normalize=True)[0]
        test_ratio = y_test.value_counts(normalize=True)[0]

        self.assertAlmostEqual(train_ratio, 0.7, delta=0.05)
        self.assertAlmostEqual(test_ratio, 0.7, delta=0.05)

    def test_split_reproducibility(self):
        """Verifica que la división sea reproducible con la misma semilla."""
        splitter1 = DataSplitter(test_size=0.3, random_state=42)
        X_train1, X_test1, y_train1, y_test1 = splitter1.split(self.test_data, "target")

        splitter2 = DataSplitter(test_size=0.3, random_state=42)
        X_train2, X_test2, y_train2, y_test2 = splitter2.split(self.test_data, "target")

        # Verificar que los splits sean idénticos
        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)
        pd.testing.assert_series_equal(y_train1, y_train2)
        pd.testing.assert_series_equal(y_test1, y_test2)


if __name__ == "__main__":
    unittest.main()

