"""
Tests para el módulo de modeling.
"""

import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from mlops_project.features import create_feature_pipeline
from mlops_project.modeling.predict import evaluate_predictions, predict
from mlops_project.modeling.train import (
    create_training_pipeline,
    get_model_instance,
    get_smote_instance,
)


class TestModelInstances(unittest.TestCase):
    """Tests para la creación de instancias de modelos."""

    def test_get_logistic_regression(self):
        """Verifica que se cree instancia de Logistic Regression."""
        model = get_model_instance("logistic_regression")

        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, "LogisticRegression")

    def test_get_decision_tree(self):
        """Verifica que se cree instancia de Decision Tree."""
        model = get_model_instance("decision_tree")

        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, "DecisionTreeClassifier")

    def test_get_random_forest(self):
        """Verifica que se cree instancia de Random Forest."""
        model = get_model_instance("random_forest")

        self.assertIsNotNone(model)
        self.assertEqual(model.__class__.__name__, "RandomForestClassifier")

    def test_get_invalid_model(self):
        """Verifica que se lance error con modelo inválido."""
        with self.assertRaises(ValueError):
            get_model_instance("nonexistent_model")

    def test_get_smote_instance(self):
        """Verifica que se cree instancia de SMOTE."""
        smote = get_smote_instance("SMOTE")

        self.assertIsNotNone(smote)
        self.assertEqual(smote.__class__.__name__, "SMOTE")

    def test_get_borderline_smote_instance(self):
        """Verifica que se cree instancia de BorderlineSMOTE."""
        smote = get_smote_instance("BorderlineSMOTE")

        self.assertIsNotNone(smote)
        self.assertEqual(smote.__class__.__name__, "BorderlineSMOTE")


class TestTrainingPipeline(unittest.TestCase):
    """Tests para la creación del pipeline de entrenamiento."""

    def setUp(self):
        """Configuración inicial para cada test."""
        self.preprocessor = create_feature_pipeline(
            include_invalid_handler=False, include_outlier_handler=False
        )
        self.model = get_model_instance("logistic_regression")

    def test_create_pipeline_with_smote(self):
        """Verifica que se cree pipeline con SMOTE."""
        pipeline = create_training_pipeline(
            preprocessor=self.preprocessor, model=self.model, use_smote=True
        )

        self.assertIsNotNone(pipeline)
        step_names = [name for name, _ in pipeline.steps]

        self.assertIn("preprocessor", step_names)
        self.assertIn("smote", step_names)
        self.assertIn("model", step_names)
        self.assertEqual(len(pipeline.steps), 3)

    def test_create_pipeline_without_smote(self):
        """Verifica que se cree pipeline sin SMOTE."""
        pipeline = create_training_pipeline(
            preprocessor=self.preprocessor, model=self.model, use_smote=False
        )

        step_names = [name for name, _ in pipeline.steps]

        self.assertIn("preprocessor", step_names)
        self.assertNotIn("smote", step_names)
        self.assertIn("model", step_names)
        self.assertEqual(len(pipeline.steps), 2)


class TestPrediction(unittest.TestCase):
    """Tests para las funciones de predicción."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear datos sintéticos simples
        np.random.seed(42)
        n_samples = 100

        # Features: 3 numéricas, 2 nominales, 2 ordinales
        self.X = pd.DataFrame(
            {
                "num1": np.random.randn(n_samples),
                "num2": np.random.randn(n_samples),
                "num3": np.random.randn(n_samples),
                "nom1": np.random.choice([1, 2, 3, 4], n_samples),
                "nom2": np.random.choice([10, 20, 30], n_samples),
                "ord1": np.random.choice([1, 2, 3], n_samples),
                "ord2": np.random.choice([1, 2, 3, 4], n_samples),
            }
        )

        # Target binario
        self.y = np.random.choice([0, 1], n_samples)

        # Crear y entrenar un modelo simple
        from mlops_project.features import FeaturePreprocessor

        preprocessor = FeaturePreprocessor(
            numeric_features=["num1", "num2", "num3"],
            nominal_features=["nom1", "nom2"],
            ordinal_features=["ord1", "ord2"],
        )

        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=preprocessor, model=model, use_smote=False
        )

        pipeline.fit(self.X, self.y)
        self.trained_pipeline = pipeline

    def test_predict_returns_classes(self):
        """Verifica que predict retorne clases."""
        predictions = predict(self.trained_pipeline, self.X, return_proba=False)

        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(len(predictions), len(self.X))
        # Verificar que solo tenga valores 0 y 1
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_predict_returns_probabilities(self):
        """Verifica que predict retorne probabilidades."""
        probabilities = predict(self.trained_pipeline, self.X, return_proba=True)

        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(len(probabilities), len(self.X))
        # Verificar que esté en rango [0, 1]
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_evaluate_predictions(self):
        """Verifica que evaluate_predictions calcule métricas."""
        y_true = self.y
        y_pred = predict(self.trained_pipeline, self.X, return_proba=False)
        y_proba = predict(self.trained_pipeline, self.X, return_proba=True)

        metrics = evaluate_predictions(y_true, y_pred, y_proba, verbose=False)

        # Verificar que se calculen todas las métricas esperadas
        expected_metrics = [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
            "confusion_matrix",
        ]

        for metric in expected_metrics:
            self.assertIn(metric, metrics)

        # Verificar rangos válidos
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)
        self.assertGreaterEqual(metrics["roc_auc"], 0)
        self.assertLessEqual(metrics["roc_auc"], 1)

    def test_evaluate_predictions_without_proba(self):
        """Verifica evaluate_predictions sin probabilidades."""
        y_true = self.y
        y_pred = predict(self.trained_pipeline, self.X, return_proba=False)

        metrics = evaluate_predictions(y_true, y_pred, y_proba=None, verbose=False)

        # Debe calcular métricas básicas
        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)

        # No debe calcular métricas que requieren probabilidades
        self.assertNotIn("roc_auc", metrics)


class TestModelPersistence(unittest.TestCase):
    """Tests para guardar y cargar modelos."""

    def setUp(self):
        """Configuración inicial para cada test."""
        # Crear directorio temporal
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = Path(self.temp_dir.name) / "test_model.joblib"

        # Crear un modelo simple
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "num1": np.random.randn(50),
                "num2": np.random.randn(50),
                "nom1": np.random.choice([1, 2, 3], 50),
                "ord1": np.random.choice([1, 2, 3], 50),
            }
        )
        y = np.random.choice([0, 1], 50)

        from mlops_project.features import FeaturePreprocessor

        preprocessor = FeaturePreprocessor(
            numeric_features=["num1", "num2"],
            nominal_features=["nom1"],
            ordinal_features=["ord1"],
        )

        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=preprocessor, model=model, use_smote=False
        )

        pipeline.fit(X, y)
        self.pipeline = pipeline
        self.X_test = X

    def tearDown(self):
        """Limpieza después de cada test."""
        self.temp_dir.cleanup()

    def test_save_and_load_model(self):
        """Verifica que se pueda guardar y cargar un modelo."""
        # Guardar modelo
        joblib.dump(self.pipeline, self.model_path)

        # Verificar que el archivo exista
        self.assertTrue(self.model_path.exists())

        # Cargar modelo
        loaded_pipeline = joblib.load(self.model_path)

        # Verificar que sea un Pipeline
        self.assertIsInstance(loaded_pipeline, Pipeline)

    def test_loaded_model_predictions_match(self):
        """Verifica que el modelo cargado haga las mismas predicciones."""
        # Predicciones del modelo original
        original_pred = self.pipeline.predict(self.X_test)

        # Guardar y cargar modelo
        joblib.dump(self.pipeline, self.model_path)
        loaded_pipeline = joblib.load(self.model_path)

        # Predicciones del modelo cargado
        loaded_pred = loaded_pipeline.predict(self.X_test)

        # Verificar que sean idénticas
        np.testing.assert_array_equal(original_pred, loaded_pred)


if __name__ == "__main__":
    unittest.main()

