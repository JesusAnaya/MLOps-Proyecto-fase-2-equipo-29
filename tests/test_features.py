"""
Tests para el módulo de features.
"""

import unittest

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlops_project.features import (
    FeaturePreprocessor,
    InvalidDataHandler,
    OutlierHandler,
    create_feature_pipeline,
)


class TestInvalidDataHandler(unittest.TestCase):
    """Tests para la clase InvalidDataHandler."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Reglas de validación simplificadas para testing
        self.validation_rules = {
            "cat1": [1, 2, 3],
            "cat2": [10, 20, 30],
        }

        # Datos de prueba con valores válidos e inválidos
        self.test_data = pd.DataFrame(
            {
                "cat1": [1, 2, 99, 3, 1],  # 99 es inválido
                "cat2": [10, 999, 20, 30, 10],  # 999 es inválido
                "num1": [1.5, 2.5, 3.5, 4.5, 5.5],  # Numérica, no se valida
            }
        )

    def test_handler_initialization(self):
        """Verifica que el handler se inicialice correctamente."""
        handler = InvalidDataHandler(validation_rules=self.validation_rules)

        self.assertEqual(handler.validation_rules, self.validation_rules)
        self.assertEqual(len(handler.mode_values_), 0)

    def test_fit_learns_mode_values(self):
        """Verifica que fit aprenda los valores de moda."""
        handler = InvalidDataHandler(validation_rules=self.validation_rules)
        handler.fit(self.test_data)

        # Debe haber aprendido modas para las columnas en validation_rules
        self.assertGreater(len(handler.mode_values_), 0)
        self.assertIn("cat1", handler.mode_values_)
        self.assertIn("cat2", handler.mode_values_)

    def test_transform_imputes_invalid_values(self):
        """Verifica que transform impute valores inválidos."""
        handler = InvalidDataHandler(validation_rules=self.validation_rules)
        handler.fit(self.test_data)
        transformed = handler.transform(self.test_data)

        # Verificar que no haya valores inválidos después de la transformación
        self.assertTrue(all(transformed["cat1"].isin(self.validation_rules["cat1"])))
        self.assertTrue(all(transformed["cat2"].isin(self.validation_rules["cat2"])))

    def test_transform_preserves_valid_values(self):
        """Verifica que transform preserve valores válidos."""
        # Datos completamente válidos
        valid_data = pd.DataFrame({"cat1": [1, 2, 3], "cat2": [10, 20, 30]})

        handler = InvalidDataHandler(validation_rules=self.validation_rules)
        handler.fit(valid_data)
        transformed = handler.transform(valid_data)

        # Los datos válidos deben permanecer sin cambios
        pd.testing.assert_frame_equal(transformed, valid_data)

    def test_fit_transform(self):
        """Verifica que fit_transform funcione correctamente."""
        handler = InvalidDataHandler(validation_rules=self.validation_rules)
        transformed = handler.fit_transform(self.test_data)

        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape, self.test_data.shape)


class TestOutlierHandler(unittest.TestCase):
    """Tests para la clase OutlierHandler."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Datos de prueba con outliers claros
        np.random.seed(42)
        normal_data = np.random.normal(50, 10, 95)
        outliers = np.array([0, 100, 150, 200, 250])  # Valores extremos

        self.test_data = pd.DataFrame(
            {"var1": np.concatenate([normal_data, outliers]), "var2": np.random.randn(100)}
        )

        self.variables = ["var1", "var2"]

    def test_handler_initialization_iqr(self):
        """Verifica que el handler IQR se inicialice correctamente."""
        handler = OutlierHandler(method="IQR", variables=self.variables)

        self.assertEqual(handler.method, "IQR")
        self.assertEqual(handler.variables, self.variables)
        self.assertEqual(len(handler.limits_), 0)

    def test_handler_initialization_percentiles(self):
        """Verifica que el handler Percentiles se inicialice correctamente."""
        handler = OutlierHandler(
            method="Percentiles", percentiles=(0.05, 0.95), variables=self.variables
        )

        self.assertEqual(handler.method, "Percentiles")
        self.assertEqual(handler.percentiles, (0.05, 0.95))

    def test_handler_invalid_method(self):
        """Verifica que se lance error con método inválido."""
        with self.assertRaises(ValueError):
            OutlierHandler(method="invalid", variables=self.variables)

    def test_fit_calculates_limits(self):
        """Verifica que fit calcule límites correctamente."""
        handler = OutlierHandler(method="IQR", variables=self.variables)
        handler.fit(self.test_data)

        # Debe haber calculado límites para ambas variables
        self.assertEqual(len(handler.limits_), 2)
        self.assertIn("var1", handler.limits_)
        self.assertIn("var2", handler.limits_)

        # Cada variable debe tener límites superior e inferior
        for var in self.variables:
            self.assertIn("lower", handler.limits_[var])
            self.assertIn("upper", handler.limits_[var])

    def test_transform_caps_outliers(self):
        """Verifica que transform delimite outliers."""
        handler = OutlierHandler(method="IQR", variables=self.variables)
        handler.fit(self.test_data)
        transformed = handler.transform(self.test_data)

        # Verificar que los valores extremos hayan sido delimitados
        self.assertLess(transformed["var1"].max(), self.test_data["var1"].max())

        # El mínimo también debe ser mayor (estamos usando valores extremos en ambos lados)
        # Pero en este caso particular, solo tenemos outliers superiores

    def test_transform_preserves_shape(self):
        """Verifica que transform preserve la forma del DataFrame."""
        handler = OutlierHandler(method="IQR", variables=self.variables)
        handler.fit(self.test_data)
        transformed = handler.transform(self.test_data)

        self.assertEqual(transformed.shape, self.test_data.shape)


class TestFeaturePreprocessor(unittest.TestCase):
    """Tests para la clase FeaturePreprocessor."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear datos de prueba con diferentes tipos de features
        np.random.seed(42)
        self.test_data = pd.DataFrame(
            {
                "num1": np.random.randn(50),
                "num2": np.random.randn(50) * 10,
                "nom1": np.random.choice([1, 2, 3, 4], 50),
                "nom2": np.random.choice([10, 20, 30], 50),
                "ord1": np.random.choice([1, 2, 3], 50),
                "ord2": np.random.choice([1, 2, 3, 4, 5], 50),
            }
        )

        self.numeric_features = ["num1", "num2"]
        self.nominal_features = ["nom1", "nom2"]
        self.ordinal_features = ["ord1", "ord2"]

    def test_preprocessor_initialization(self):
        """Verifica que el preprocessor se inicialice correctamente."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        self.assertEqual(preprocessor.numeric_features, self.numeric_features)
        self.assertEqual(preprocessor.nominal_features, self.nominal_features)
        self.assertEqual(preprocessor.ordinal_features, self.ordinal_features)
        self.assertIsNone(preprocessor.preprocessor_)

    def test_fit_creates_preprocessor(self):
        """Verifica que fit cree el preprocessor interno."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        preprocessor.fit(self.test_data)

        self.assertIsNotNone(preprocessor.preprocessor_)

    def test_transform_changes_shape(self):
        """Verifica que transform cambie la forma (por OHE)."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        preprocessor.fit(self.test_data)
        transformed = preprocessor.transform(self.test_data)

        # El número de columnas debe aumentar debido a One-Hot Encoding
        self.assertGreater(transformed.shape[1], self.test_data.shape[1])
        # El número de filas debe mantenerse igual
        self.assertEqual(transformed.shape[0], self.test_data.shape[0])

    def test_transform_before_fit_raises_error(self):
        """Verifica que transform sin fit lance error."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        with self.assertRaises(RuntimeError):
            preprocessor.transform(self.test_data)

    def test_fit_transform(self):
        """Verifica que fit_transform funcione correctamente."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        transformed = preprocessor.fit_transform(self.test_data)

        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(transformed.shape[0], self.test_data.shape[0])

    def test_get_feature_names_out(self):
        """Verifica que get_feature_names_out funcione."""
        preprocessor = FeaturePreprocessor(
            numeric_features=self.numeric_features,
            nominal_features=self.nominal_features,
            ordinal_features=self.ordinal_features,
        )

        preprocessor.fit(self.test_data)
        feature_names = preprocessor.get_feature_names_out()

        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)


class TestFeaturePipeline(unittest.TestCase):
    """Tests para la creación del pipeline de features."""

    def test_create_pipeline_full(self):
        """Verifica que se cree el pipeline completo."""
        pipeline = create_feature_pipeline(
            include_invalid_handler=True, include_outlier_handler=True
        )

        self.assertIsInstance(pipeline, Pipeline)
        self.assertEqual(len(pipeline.steps), 3)  # invalid, outlier, preprocessor

        # Verificar nombres de los steps
        step_names = [name for name, _ in pipeline.steps]
        self.assertIn("invalid_handler", step_names)
        self.assertIn("outlier_handler", step_names)
        self.assertIn("feature_preprocessor", step_names)

    def test_create_pipeline_without_invalid_handler(self):
        """Verifica pipeline sin invalid handler."""
        pipeline = create_feature_pipeline(
            include_invalid_handler=False, include_outlier_handler=True
        )

        step_names = [name for name, _ in pipeline.steps]
        self.assertNotIn("invalid_handler", step_names)
        self.assertIn("outlier_handler", step_names)
        self.assertIn("feature_preprocessor", step_names)

    def test_create_pipeline_without_outlier_handler(self):
        """Verifica pipeline sin outlier handler."""
        pipeline = create_feature_pipeline(
            include_invalid_handler=True, include_outlier_handler=False
        )

        step_names = [name for name, _ in pipeline.steps]
        self.assertIn("invalid_handler", step_names)
        self.assertNotIn("outlier_handler", step_names)
        self.assertIn("feature_preprocessor", step_names)

    def test_create_pipeline_minimal(self):
        """Verifica pipeline mínimo (solo preprocessor)."""
        pipeline = create_feature_pipeline(
            include_invalid_handler=False, include_outlier_handler=False
        )

        self.assertEqual(len(pipeline.steps), 1)
        step_names = [name for name, _ in pipeline.steps]
        self.assertIn("feature_preprocessor", step_names)


if __name__ == "__main__":
    unittest.main()

