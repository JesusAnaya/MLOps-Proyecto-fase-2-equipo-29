"""
Tests mejorados para el módulo de modeling usando mocks.

Este módulo usa pytest-mock para evitar entrenar modelos reales,
enfocándose en probar la lógica del pipeline MLOps.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlops_project.modeling.predict import (
    batch_predict,
    evaluate_predictions,
    load_model,
    predict,
)
from mlops_project.modeling.train import (
    create_training_pipeline,
    evaluate_model,
    get_model_instance,
    get_smote_instance,
    save_results,
    train_model,
)


class TestModelInstancesImproved:
    """Tests mejorados para creación de instancias de modelos."""

    def test_get_logistic_regression(self):
        """Verifica que se cree instancia de Logistic Regression."""
        model = get_model_instance("logistic_regression")

        assert model is not None
        assert model.__class__.__name__ == "LogisticRegression"
        # Verificar que tenga los parámetros correctos
        assert model.penalty == "l2"
        assert model.solver == "newton-cg"

    def test_get_decision_tree(self):
        """Verifica que se cree instancia de Decision Tree."""
        model = get_model_instance("decision_tree")

        assert model is not None
        assert model.__class__.__name__ == "DecisionTreeClassifier"
        assert model.max_depth == 3

    def test_get_random_forest(self):
        """Verifica que se cree instancia de Random Forest."""
        model = get_model_instance("random_forest")

        assert model is not None
        assert model.__class__.__name__ == "RandomForestClassifier"
        assert model.n_estimators == 200

    def test_get_invalid_model(self):
        """Verifica que se lance error con modelo inválido."""
        with pytest.raises(ValueError, match="no disponible"):
            get_model_instance("nonexistent_model")

    def test_get_smote_instance(self):
        """Verifica que se cree instancia de SMOTE."""
        smote = get_smote_instance("SMOTE")

        assert smote is not None
        assert smote.__class__.__name__ == "SMOTE"

    def test_get_borderline_smote_instance(self):
        """Verifica que se cree instancia de BorderlineSMOTE."""
        smote = get_smote_instance("BorderlineSMOTE")

        assert smote is not None
        assert smote.__class__.__name__ == "BorderlineSMOTE"


class TestTrainingPipelineImproved:
    """Tests mejorados para creación del pipeline de entrenamiento."""

    def test_create_pipeline_with_smote(self, mock_preprocessor):
        """Verifica que se cree pipeline con SMOTE."""
        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=mock_preprocessor, model=model, use_smote=True
        )

        assert pipeline is not None
        step_names = [name for name, _ in pipeline.steps]

        assert "preprocessor" in step_names
        assert "smote" in step_names
        assert "model" in step_names
        assert len(pipeline.steps) == 3

    def test_create_pipeline_without_smote(self, mock_preprocessor):
        """Verifica que se cree pipeline sin SMOTE."""
        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=mock_preprocessor, model=model, use_smote=False
        )

        step_names = [name for name, _ in pipeline.steps]

        assert "preprocessor" in step_names
        assert "smote" not in step_names
        assert "model" in step_names
        assert len(pipeline.steps) == 2

    def test_pipeline_has_correct_model(self, mock_preprocessor):
        """Verifica que el pipeline contenga el modelo correcto."""
        model = get_model_instance("random_forest")

        pipeline = create_training_pipeline(
            preprocessor=mock_preprocessor, model=model, use_smote=False
        )

        # Obtener el modelo del pipeline
        final_model = pipeline.named_steps["model"]

        assert final_model.__class__.__name__ == "RandomForestClassifier"


class TestEvaluateModelMocked:
    """Tests para evaluate_model usando mocks."""

    @patch("mlops_project.modeling.train.cross_validate")
    def test_evaluate_model_with_mock(
        self, mock_cross_validate, mock_preprocessor, sample_X_y, sample_cv_results
    ):
        """Verifica evaluate_model usando mock de cross_validate."""
        X, y = sample_X_y
        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=mock_preprocessor, model=model, use_smote=False
        )

        # Configurar mock de cross_validate
        mock_cross_validate.return_value = sample_cv_results

        # Ejecutar evaluación
        results = evaluate_model(pipeline, X, y, cv_folds=5, cv_repeats=3, verbose=False)

        # Verificar que cross_validate fue llamado
        assert mock_cross_validate.called

        # Verificar estructura de resultados
        assert isinstance(results, dict)
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results

        # Verificar que cada métrica tiene test_mean y test_std
        for metric_name, scores in results.items():
            assert "test_mean" in scores
            assert "test_std" in scores
            assert "train_mean" in scores
            assert "train_std" in scores

            # Verificar que son floats
            assert isinstance(scores["test_mean"], float)
            assert isinstance(scores["test_std"], float)

    @patch("mlops_project.modeling.train.cross_validate")
    def test_evaluate_model_calculates_correct_stats(
        self, mock_cross_validate, mock_preprocessor, sample_X_y, sample_cv_results
    ):
        """Verifica que evaluate_model calcule estadísticas correctamente."""
        X, y = sample_X_y
        model = get_model_instance("logistic_regression")

        pipeline = create_training_pipeline(
            preprocessor=mock_preprocessor, model=model, use_smote=False
        )

        # Configurar mock con valores conocidos
        mock_cv_results = {
            "test_accuracy": np.array([0.8, 0.85, 0.9]),
            "train_accuracy": np.array([0.82, 0.87, 0.92]),
        }
        mock_cross_validate.return_value = mock_cv_results

        results = evaluate_model(pipeline, X, y, cv_folds=3, cv_repeats=1, verbose=False)

        # Verificar cálculos
        assert results["accuracy"]["test_mean"] == pytest.approx(0.85, rel=1e-5)
        assert results["accuracy"]["test_std"] == pytest.approx(
            np.std([0.8, 0.85, 0.9]), rel=1e-5
        )


class TestTrainModelMocked:
    """Tests para train_model usando mocks."""

    @patch("mlops_project.modeling.train.evaluate_model")
    @patch("mlops_project.modeling.train.joblib.dump")
    def test_train_model_without_evaluation(
        self, mock_dump, mock_evaluate, mock_preprocessor, sample_X_y
    ):
        """Verifica train_model sin evaluación."""
        X, y = sample_X_y

        # Mock evaluate_model para que no se ejecute cross-validation real
        mock_evaluate.return_value = {"accuracy": {"test_mean": 0.8, "test_std": 0.05}}

        pipeline, results = train_model(
            X_train=X,
            y_train=y,
            preprocessor=mock_preprocessor,
            model_name="logistic_regression",
            use_smote=False,
            evaluate=False,
            save_model=False,
        )

        # Verificar que no se llamó evaluate
        assert not mock_evaluate.called

        # Verificar que no se guardó el modelo
        assert not mock_dump.called

        # Verificar que results es None
        assert results is None

        # Verificar que se retorna un pipeline
        assert pipeline is not None

    @patch("mlops_project.modeling.train.evaluate_model")
    @patch("mlops_project.modeling.train.joblib.dump")
    def test_train_model_with_evaluation(
        self, mock_dump, mock_evaluate, mock_preprocessor, sample_X_y
    ):
        """Verifica train_model con evaluación."""
        X, y = sample_X_y

        mock_eval_results = {"accuracy": {"test_mean": 0.8, "test_std": 0.05}}
        mock_evaluate.return_value = mock_eval_results

        pipeline, results = train_model(
            X_train=X,
            y_train=y,
            preprocessor=mock_preprocessor,
            model_name="logistic_regression",
            use_smote=False,
            evaluate=True,
            save_model=False,
        )

        # Verificar que se llamó evaluate
        assert mock_evaluate.called

        # Verificar que results contiene los resultados
        assert results == mock_eval_results

    @patch("mlops_project.modeling.train.joblib.dump")
    def test_train_model_saves_when_requested(
        self, mock_dump, mock_preprocessor, sample_X_y, mocker
    ):
        """Verifica que train_model guarde el modelo cuando se solicita."""
        X, y = sample_X_y

        # Mock get_model_path
        mock_path = mocker.patch("mlops_project.modeling.train.get_model_path")
        mock_path.return_value = Path("/tmp/test_model.joblib")

        train_model(
            X_train=X,
            y_train=y,
            preprocessor=mock_preprocessor,
            model_name="logistic_regression",
            use_smote=False,
            evaluate=False,
            save_model=True,
            model_filename="test_model.joblib",
        )

        # Verificar que se llamó joblib.dump
        assert mock_dump.called
        # Verificar que se llamó con el path correcto
        assert mock_path.called


class TestPredictionMocked:
    """Tests mejorados para predicción usando mocks."""

    def test_predict_returns_classes(self, mock_trained_model, sample_X_y):
        """Verifica que predict retorne clases."""
        X, _ = sample_X_y

        predictions = predict(mock_trained_model, X, return_proba=False)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert np.all(np.isin(predictions, [0, 1]))

        # Verificar que se llamó predict
        mock_trained_model.predict.assert_called_once()

    def test_predict_returns_probabilities(self, mock_trained_model, sample_X_y):
        """Verifica que predict retorne probabilidades."""
        X, _ = sample_X_y

        probabilities = predict(mock_trained_model, X, return_proba=True)

        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(X)
        assert np.all((probabilities >= 0) & (probabilities <= 1))

        # Verificar que se llamó predict_proba
        mock_trained_model.predict_proba.assert_called_once()

    def test_predict_raises_error_without_predict_proba(self, mocker, sample_X_y):
        """Verifica que predict lance error si el modelo no tiene predict_proba."""
        X, _ = sample_X_y

        # Crear modelo mock sin predict_proba
        mock_model = mocker.MagicMock(spec=["predict"])  # Solo tiene predict

        with pytest.raises(AttributeError, match="no soporta predict_proba"):
            predict(mock_model, X, return_proba=True)


class TestEvaluatePredictions:
    """Tests para evaluate_predictions (no necesita mocks)."""

    def test_evaluate_predictions_basic_metrics(self, sample_predictions):
        """Verifica que evaluate_predictions calcule métricas básicas."""
        y_true, y_pred, y_proba = sample_predictions

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
            assert metric in metrics

        # Verificar rangos válidos
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_evaluate_predictions_without_proba(self, sample_predictions):
        """Verifica evaluate_predictions sin probabilidades."""
        y_true, y_pred, _ = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred, y_proba=None, verbose=False)

        # Debe calcular métricas básicas
        assert "accuracy" in metrics
        assert "precision" in metrics

        # No debe calcular métricas que requieren probabilidades
        assert "roc_auc" not in metrics

    def test_confusion_matrix_structure(self, sample_predictions):
        """Verifica estructura de la matriz de confusión."""
        y_true, y_pred, y_proba = sample_predictions

        metrics = evaluate_predictions(y_true, y_pred, y_proba, verbose=False)

        cm = metrics["confusion_matrix"]

        # Verificar que tenga las claves correctas
        assert "true_negatives" in cm
        assert "false_positives" in cm
        assert "false_negatives" in cm
        assert "true_positives" in cm

        # Verificar que sean enteros
        assert isinstance(cm["true_negatives"], int)
        assert isinstance(cm["true_positives"], int)


class TestBatchPredict:
    """Tests para batch_predict."""

    def test_batch_predict_processes_in_batches(self, mock_trained_model):
        """Verifica que batch_predict procese en batches."""
        X = pd.DataFrame(np.random.randn(250, 5))  # 250 muestras
        batch_size = 100

        predictions = batch_predict(mock_trained_model, X, batch_size=batch_size)

        # Debe haber llamado predict 3 veces (250/100 = 3 batches)
        assert mock_trained_model.predict.call_count == 3

        # Debe retornar todas las predicciones
        assert len(predictions) == 250

    def test_batch_predict_with_probabilities(self, mock_trained_model):
        """Verifica batch_predict con probabilidades."""
        X = pd.DataFrame(np.random.randn(150, 5))
        batch_size = 50

        probabilities = batch_predict(
            mock_trained_model, X, batch_size=batch_size, return_proba=True
        )

        # Debe haber llamado predict_proba 3 veces
        assert mock_trained_model.predict_proba.call_count == 3

        # Debe retornar todas las probabilidades
        assert len(probabilities) == 150
        assert np.all((probabilities >= 0) & (probabilities <= 1))


class TestModelPersistenceMocked:
    """Tests para persistencia de modelos con mocks."""

    def test_save_and_load_model_mock(self, mocker):
        """Verifica guardar y cargar modelo usando mocks."""
        # Crear directorio temporal
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.joblib"

            # Crear modelo mock simple
            mock_model = mocker.MagicMock()
            mock_model.predict.return_value = np.array([0, 1, 0])

            # Guardar modelo
            joblib.dump(mock_model, model_path)

            # Verificar que el archivo exista
            assert model_path.exists()

            # Cargar modelo
            loaded_model = joblib.load(model_path)

            # Verificar que se pueda usar
            predictions = loaded_model.predict(None)
            assert len(predictions) == 3

    @patch("mlops_project.modeling.predict.joblib.load")
    @patch("mlops_project.modeling.predict.get_model_path")
    def test_load_model_calls_correct_functions(self, mock_get_path, mock_load, mocker):
        """Verifica que load_model llame las funciones correctas."""
        # Configurar mocks
        mock_path = Path("/fake/path/model.joblib")
        mock_get_path.return_value = mock_path
        mock_model = mocker.MagicMock()
        mock_load.return_value = mock_model

        # Simular que el archivo existe
        with patch.object(Path, "exists", return_value=True):
            model = load_model("test_model.joblib")

        # Verificar llamadas
        mock_get_path.assert_called_once_with("test_model.joblib")
        mock_load.assert_called_once_with(mock_path)

        # Verificar retorno
        assert model == mock_model


class TestSaveResults:
    """Tests para save_results."""

    @patch("mlops_project.modeling.train.get_model_path")
    @patch("builtins.open", create=True)
    def test_save_results_creates_json(self, mock_open, mock_get_path, mocker):
        """Verifica que save_results cree archivo JSON correctamente."""
        mock_path = Path("/fake/path/results.json")
        mock_get_path.return_value = mock_path

        # Mock file handle
        mock_file = mocker.MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        results = {
            "accuracy": {"test_mean": 0.8, "test_std": 0.05},
            "precision": {"test_mean": 0.82, "test_std": 0.04},
        }

        save_results(results, "logistic_regression")

        # Verificar que se llamó open
        mock_open.assert_called_once()

        # Verificar que se intentó escribir
        mock_file.write.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

