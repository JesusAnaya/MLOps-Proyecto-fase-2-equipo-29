"""
Configuración de fixtures compartidas para pytest.

Este módulo define fixtures reutilizables para todos los tests.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data_df():
    """
    Fixture que proporciona un DataFrame de muestra para tests.

    Returns:
        pd.DataFrame: DataFrame con datos sintéticos para testing
    """
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "num1": np.random.randn(n_samples),
            "num2": np.random.randn(n_samples) * 10,
            "num3": np.random.randn(n_samples),
            "nom1": np.random.choice([1, 2, 3, 4], n_samples),
            "nom2": np.random.choice([10, 20, 30], n_samples),
            "ord1": np.random.choice([1, 2, 3], n_samples),
            "ord2": np.random.choice([1, 2, 3, 4], n_samples),
            "target": np.random.choice([0, 1], n_samples),
        }
    )


@pytest.fixture
def sample_X_y(sample_data_df):
    """
    Fixture que proporciona X y y separados para tests.

    Args:
        sample_data_df: DataFrame de muestra

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features y target
    """
    y = sample_data_df["target"].copy()
    X = sample_data_df.drop(columns=["target"]).copy()
    return X, y


@pytest.fixture
def sample_feature_names():
    """
    Fixture que proporciona nombres de features por tipo.

    Returns:
        Dict: Diccionario con listas de features por tipo
    """
    return {
        "numeric": ["num1", "num2", "num3"],
        "nominal": ["nom1", "nom2"],
        "ordinal": ["ord1", "ord2"],
    }


@pytest.fixture
def mock_trained_model(mocker):
    """
    Fixture que proporciona un modelo mock pre-entrenado.

    Args:
        mocker: fixture de pytest-mock

    Returns:
        MagicMock: Modelo mock con métodos predict y predict_proba
    """
    mock_model = mocker.MagicMock()

    # Configurar comportamiento del mock
    def predict_side_effect(X):
        return np.random.choice([0, 1], size=len(X))

    def predict_proba_side_effect(X):
        proba_class_1 = np.random.rand(len(X))
        proba_class_0 = 1 - proba_class_1
        return np.column_stack([proba_class_0, proba_class_1])

    mock_model.predict.side_effect = predict_side_effect
    mock_model.predict_proba.side_effect = predict_proba_side_effect

    return mock_model


@pytest.fixture
def mock_preprocessor(mocker):
    """
    Fixture que proporciona un preprocessor mock.

    Args:
        mocker: fixture de pytest-mock

    Returns:
        MagicMock: Preprocessor mock con métodos fit y transform
    """
    mock_prep = mocker.MagicMock()

    # Configurar comportamiento del mock
    def transform_side_effect(X):
        # Simular transformación aumentando columnas (como OHE)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(np.random.randn(len(X), 15))
        return np.random.randn(len(X), 15)

    mock_prep.fit.return_value = mock_prep
    mock_prep.transform.side_effect = transform_side_effect
    mock_prep.fit_transform.side_effect = transform_side_effect

    return mock_prep


@pytest.fixture
def sample_cv_results():
    """
    Fixture que proporciona resultados mock de cross-validation.

    Returns:
        Dict: Resultados simulados de cross_validate
    """
    n_splits = 15  # 5 folds x 3 repeats
    return {
        "test_accuracy": np.random.uniform(0.7, 0.85, n_splits),
        "train_accuracy": np.random.uniform(0.75, 0.9, n_splits),
        "test_precision": np.random.uniform(0.7, 0.85, n_splits),
        "train_precision": np.random.uniform(0.75, 0.9, n_splits),
        "test_recall": np.random.uniform(0.65, 0.8, n_splits),
        "train_recall": np.random.uniform(0.7, 0.85, n_splits),
        "test_f1": np.random.uniform(0.7, 0.82, n_splits),
        "train_f1": np.random.uniform(0.72, 0.87, n_splits),
        "test_roc_auc": np.random.uniform(0.75, 0.88, n_splits),
        "train_roc_auc": np.random.uniform(0.8, 0.92, n_splits),
        "test_average_precision": np.random.uniform(0.75, 0.88, n_splits),
        "train_average_precision": np.random.uniform(0.8, 0.92, n_splits),
        "test_geometric_mean": np.random.uniform(0.7, 0.82, n_splits),
        "train_geometric_mean": np.random.uniform(0.72, 0.87, n_splits),
    }


@pytest.fixture
def sample_predictions():
    """
    Fixture que proporciona predicciones y probabilidades de muestra.

    Returns:
        Tuple: (y_true, y_pred, y_proba)
    """
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.choice([0, 1], n_samples)
    y_pred = np.random.choice([0, 1], n_samples)
    y_proba = np.random.rand(n_samples)

    return y_true, y_pred, y_proba

