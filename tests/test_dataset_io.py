"""
Tests adicionales para dataset con mocks de I/O.

Este módulo agrega tests para las funciones de I/O usando mocks.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from mlops_project.dataset import (
    DataCleaner,
    DataLoader,
    DataSplitter,
    load_and_prepare_data,
)


class TestDataLoaderIO:
    """Tests para DataLoader con mocks de I/O."""

    @patch("mlops_project.dataset.pd.read_csv")
    def test_load_data_calls_read_csv(self, mock_read_csv, sample_data_df):
        """Verifica que load_data llame pd.read_csv correctamente."""
        mock_read_csv.return_value = sample_data_df

        loader = DataLoader("fake_path.csv")
        data = loader.load_data()

        # Verificar que se llamó read_csv
        mock_read_csv.assert_called_once_with("fake_path.csv", sep=",")

        # Verificar que retorne el DataFrame
        assert data is not None
        assert len(data) == len(sample_data_df)

    @patch("mlops_project.dataset.pd.read_csv")
    def test_load_data_handles_file_not_found(self, mock_read_csv):
        """Verifica manejo de archivo no encontrado."""
        mock_read_csv.side_effect = FileNotFoundError("File not found")

        loader = DataLoader("nonexistent.csv")
        data = loader.load_data()

        # Debe retornar None
        assert data is None

    @patch("mlops_project.dataset.pd.read_csv")
    def test_load_data_handles_generic_exception(self, mock_read_csv):
        """Verifica manejo de excepciones genéricas."""
        mock_read_csv.side_effect = Exception("Generic error")

        loader = DataLoader("bad_file.csv")
        data = loader.load_data()

        # Debe retornar None
        assert data is None

    @patch("mlops_project.dataset.pd.read_csv")
    def test_load_data_with_custom_delimiter(self, mock_read_csv, sample_data_df):
        """Verifica carga con delimitador personalizado."""
        mock_read_csv.return_value = sample_data_df

        loader = DataLoader("file.tsv", delimiter="\t")
        loader.load_data()

        # Verificar que se pasó el delimitador correcto
        mock_read_csv.assert_called_once_with("file.tsv", sep="\t")


class TestDataCleanerAdvanced:
    """Tests avanzados para DataCleaner."""

    def test_clean_data_handles_all_invalid_target(self):
        """Verifica comportamiento cuando todos los targets son inválidos."""
        df = pd.DataFrame({"feature1": [1, 2, 3], "kredit": [np.nan, 2, 3]})

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(df)

        # Solo la primera fila tiene target válido (después de conversión a NaN)
        assert len(cleaned) == 0  # Todas las filas tienen target inválido

    def test_clean_data_preserves_valid_rows(self):
        """Verifica que se preserven filas válidas."""
        df = pd.DataFrame(
            {"feature1": [1, 2, 3, 4], "kredit": [0, 1, 0, 1], "mixed_type_col": list("abcd")}
        )

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(df)

        # Todas las filas tienen target válido
        assert len(cleaned) == 4
        assert "mixed_type_col" not in cleaned.columns

    def test_convert_to_numeric_handles_mixed_types(self):
        """Verifica conversión de tipos mixtos."""
        df = pd.DataFrame(
            {"col1": [1, "invalid", 3, "error", 5], "col2": [10, 20, "bad", 40, 50]}
        )

        cleaner = DataCleaner(target_column="col1", mixed_type_column=None)
        cleaned = cleaner._convert_to_numeric(df)

        # Verificar que las columnas son numéricas
        assert pd.api.types.is_numeric_dtype(cleaned["col1"])
        assert pd.api.types.is_numeric_dtype(cleaned["col2"])

        # Verificar que valores inválidos son NaN
        assert cleaned["col1"].isna().sum() == 2
        assert cleaned["col2"].isna().sum() == 1


class TestDataSplitterAdvanced:
    """Tests avanzados para DataSplitter."""

    def test_split_with_unbalanced_data(self):
        """Verifica división con datos desbalanceados."""
        # Crear datos muy desbalanceados (90% clase 0, 10% clase 1)
        df = pd.DataFrame(
            {"feature1": np.random.randn(100), "target": [0] * 90 + [1] * 10}
        )

        splitter = DataSplitter(test_size=0.2, stratify=True)
        X_train, X_test, y_train, y_test = splitter.split(df, "target")

        # Verificar que mantiene aproximadamente la proporción
        train_ratio = (y_train == 0).sum() / len(y_train)
        test_ratio = (y_test == 0).sum() / len(y_test)

        assert 0.85 <= train_ratio <= 0.95
        assert 0.85 <= test_ratio <= 0.95

    def test_split_without_stratification(self):
        """Verifica división sin estratificación."""
        df = pd.DataFrame({"feature1": np.random.randn(100), "target": [0] * 70 + [1] * 30})

        splitter = DataSplitter(test_size=0.3, stratify=False)
        X_train, X_test, y_train, y_test = splitter.split(df, "target")

        # Debe dividir correctamente aunque sin garantía de proporciones exactas
        assert len(X_train) + len(X_test) == 100

    def test_split_with_different_test_sizes(self):
        """Verifica diferentes tamaños de test."""
        df = pd.DataFrame({"feature1": np.random.randn(100), "target": np.random.choice([0, 1], 100)})

        for test_size in [0.1, 0.2, 0.3, 0.4]:
            splitter = DataSplitter(test_size=test_size)
            X_train, X_test, y_train, y_test = splitter.split(df, "target")

            expected_test = int(100 * test_size)
            assert abs(len(X_test) - expected_test) <= 2  # Tolerancia de ±2


class TestLoadAndPrepareDataMocked:
    """Tests para load_and_prepare_data con mocks."""

    @patch("mlops_project.dataset.DataLoader")
    @patch("mlops_project.dataset.DataCleaner")
    @patch("mlops_project.dataset.DataSplitter")
    def test_load_and_prepare_data_calls_components(
        self, mock_splitter_class, mock_cleaner_class, mock_loader_class, sample_data_df
    ):
        """Verifica que load_and_prepare_data llame todos los componentes."""
        # Configurar mocks
        mock_loader = MagicMock()
        mock_loader.load_data.return_value = sample_data_df
        mock_loader_class.return_value = mock_loader

        mock_cleaner = MagicMock()
        mock_cleaner.clean_data.return_value = sample_data_df
        mock_cleaner_class.return_value = mock_cleaner

        mock_splitter = MagicMock()
        X_train = sample_data_df.drop(columns=["target"])
        y_train = sample_data_df["target"]
        mock_splitter.split.return_value = (X_train, X_train, y_train, y_train)
        mock_splitter_class.return_value = mock_splitter

        # Ejecutar función
        result = load_and_prepare_data("fake_path.csv", save_processed=False)

        # Verificar llamadas
        mock_loader_class.assert_called_once()
        mock_loader.load_data.assert_called_once()
        mock_cleaner_class.assert_called_once()
        mock_cleaner.clean_data.assert_called_once()
        mock_splitter_class.assert_called_once()
        mock_splitter.split.assert_called_once()

        # Verificar resultado
        assert result is not None
        assert len(result) == 4  # X_train, X_test, y_train, y_test

    @patch("mlops_project.dataset.DataLoader")
    def test_load_and_prepare_data_handles_load_failure(self, mock_loader_class):
        """Verifica manejo de fallo en carga."""
        # Configurar loader para retornar None
        mock_loader = MagicMock()
        mock_loader.load_data.return_value = None
        mock_loader_class.return_value = mock_loader

        # Debe lanzar RuntimeError
        with pytest.raises(RuntimeError, match="No se pudieron cargar los datos"):
            load_and_prepare_data("fake_path.csv")

    @patch("mlops_project.dataset.DataLoader")
    @patch("mlops_project.dataset.DataCleaner")
    @patch("mlops_project.dataset.pd.DataFrame.to_csv")
    def test_load_and_prepare_data_saves_when_requested(
        self, mock_to_csv, mock_cleaner_class, mock_loader_class, sample_data_df
    ):
        """Verifica que guarde cuando se solicita."""
        # Configurar mocks
        mock_loader = MagicMock()
        mock_loader.load_data.return_value = sample_data_df
        mock_loader_class.return_value = mock_loader

        mock_cleaner = MagicMock()
        mock_cleaner.clean_data.return_value = sample_data_df
        mock_cleaner_class.return_value = mock_cleaner

        # Ejecutar con save=True
        load_and_prepare_data("fake_path.csv", save_processed=True, return_combined=True)

        # Verificar que se llamó to_csv al menos una vez
        assert mock_to_csv.called

    @patch("mlops_project.dataset.DataLoader")
    @patch("mlops_project.dataset.DataCleaner")
    def test_load_and_prepare_data_return_combined(
        self, mock_cleaner_class, mock_loader_class, sample_data_df
    ):
        """Verifica modo return_combined."""
        # Configurar mocks
        mock_loader = MagicMock()
        mock_loader.load_data.return_value = sample_data_df
        mock_loader_class.return_value = mock_loader

        mock_cleaner = MagicMock()
        mock_cleaner.clean_data.return_value = sample_data_df
        mock_cleaner_class.return_value = mock_cleaner

        # Ejecutar con return_combined=True
        result = load_and_prepare_data(
            "fake_path.csv", save_processed=False, return_combined=True
        )

        # Debe retornar X, y (no divididos)
        assert len(result) == 2
        X, y = result
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)


class TestDataCleanerEdgeCases:
    """Tests para casos extremos de DataCleaner."""

    def test_clean_data_empty_dataframe(self):
        """Verifica comportamiento con DataFrame vacío."""
        df = pd.DataFrame()

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(df)

        # Debe retornar DataFrame vacío sin errores
        assert len(cleaned) == 0

    def test_clean_data_all_nulls(self):
        """Verifica comportamiento con solo valores nulos."""
        df = pd.DataFrame(
            {"feature1": [np.nan, np.nan, np.nan], "kredit": [np.nan, np.nan, np.nan]}
        )

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(df)

        # Debe eliminar todas las filas
        assert len(cleaned) == 0

    def test_clean_data_single_row(self):
        """Verifica comportamiento con una sola fila."""
        df = pd.DataFrame({"feature1": [42], "kredit": [1], "mixed_type_col": ["a"]})

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(df)

        # Debe procesar correctamente
        assert len(cleaned) == 1
        assert "mixed_type_col" not in cleaned.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

