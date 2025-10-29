"""
Tests para el módulo de configuración.
"""

import unittest
from pathlib import Path

from mlops_project import config


class TestConfig(unittest.TestCase):
    """Tests para verificar la configuración del proyecto."""

    def test_project_root_exists(self):
        """Verifica que la ruta del proyecto exista."""
        self.assertTrue(config.PROJECT_ROOT.exists())
        self.assertTrue(config.PROJECT_ROOT.is_dir())

    def test_data_directories_created(self):
        """Verifica que los directorios de datos se hayan creado."""
        self.assertTrue(config.DATA_DIR.exists())
        self.assertTrue(config.RAW_DATA_DIR.exists())
        self.assertTrue(config.PROCESSED_DATA_DIR.exists())
        self.assertTrue(config.INTERIM_DATA_DIR.exists())

    def test_model_directory_created(self):
        """Verifica que el directorio de modelos exista."""
        self.assertTrue(config.MODELS_DIR.exists())

    def test_reports_directory_created(self):
        """Verifica que el directorio de reportes exista."""
        self.assertTrue(config.REPORTS_DIR.exists())
        self.assertTrue(config.FIGURES_DIR.exists())

    def test_random_seed_is_int(self):
        """Verifica que la semilla aleatoria sea un entero."""
        self.assertIsInstance(config.RANDOM_SEED, int)
        self.assertEqual(config.RANDOM_SEED, 42)

    def test_target_column_defined(self):
        """Verifica que la columna objetivo esté definida."""
        self.assertIsInstance(config.TARGET_COLUMN, str)
        self.assertEqual(config.TARGET_COLUMN, "kredit")

    def test_feature_lists_not_empty(self):
        """Verifica que las listas de features no estén vacías."""
        self.assertGreater(len(config.NUMERIC_FEATURES), 0)
        self.assertGreater(len(config.ORDINAL_FEATURES), 0)
        self.assertGreater(len(config.NOMINAL_FEATURES), 0)

    def test_feature_lists_no_duplicates(self):
        """Verifica que no haya duplicados entre las listas de features."""
        all_features = (
            config.NUMERIC_FEATURES + config.ORDINAL_FEATURES + config.NOMINAL_FEATURES
        )
        self.assertEqual(len(all_features), len(set(all_features)))

    def test_validation_rules_not_empty(self):
        """Verifica que las reglas de validación estén definidas."""
        self.assertGreater(len(config.CATEGORICAL_VALIDATION_RULES), 0)
        self.assertIn("kredit", config.CATEGORICAL_VALIDATION_RULES)

    def test_test_size_valid_range(self):
        """Verifica que test_size esté en el rango válido."""
        self.assertGreater(config.TEST_SIZE, 0)
        self.assertLess(config.TEST_SIZE, 1)

    def test_cv_config_valid(self):
        """Verifica que la configuración de cross-validation sea válida."""
        self.assertGreater(config.CV_FOLDS, 1)
        self.assertGreater(config.CV_REPEATS, 0)

    def test_available_models_defined(self):
        """Verifica que haya modelos disponibles."""
        self.assertGreater(len(config.AVAILABLE_MODELS), 0)
        self.assertIn("logistic_regression", config.AVAILABLE_MODELS)

    def test_get_data_path(self):
        """Verifica que get_data_path funcione correctamente."""
        path = config.get_data_path("test.csv", "raw")
        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith("test.csv"))

    def test_get_data_path_invalid_type(self):
        """Verifica que get_data_path lance error con tipo inválido."""
        with self.assertRaises(ValueError):
            config.get_data_path("test.csv", "invalid")

    def test_get_model_path(self):
        """Verifica que get_model_path funcione correctamente."""
        path = config.get_model_path("model.joblib")
        self.assertIsInstance(path, Path)
        self.assertTrue(str(path).endswith("model.joblib"))

    def test_outlier_config(self):
        """Verifica que la configuración de outliers sea válida."""
        self.assertIn(config.OUTLIER_METHOD, ["IQR", "Percentiles"])
        self.assertIsInstance(config.OUTLIER_VARIABLES, list)
        self.assertGreater(len(config.OUTLIER_VARIABLES), 0)

    def test_numeric_scaler_range(self):
        """Verifica que el rango del scaler sea válido."""
        self.assertIsInstance(config.NUMERIC_SCALER_RANGE, tuple)
        self.assertEqual(len(config.NUMERIC_SCALER_RANGE), 2)
        self.assertLess(config.NUMERIC_SCALER_RANGE[0], config.NUMERIC_SCALER_RANGE[1])


if __name__ == "__main__":
    unittest.main()

