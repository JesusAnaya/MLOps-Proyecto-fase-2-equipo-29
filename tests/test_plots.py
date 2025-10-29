"""
Tests para el módulo de plots usando mocks.

Este módulo usa mocks para evitar crear archivos y mostrar gráficos reales.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mlops_project import plots


class TestPlotDistribution:
    """Tests para plot_distribution."""

    @patch("mlops_project.plots.plt")
    def test_plot_distribution_basic(self, mock_plt, sample_data_df):
        """Verifica que plot_distribution cree subplots correctamente."""
        plots.plot_distribution(sample_data_df, save_path=None)

        # Verificar que se llamó subplots
        mock_plt.subplots.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    def test_plot_distribution_saves_file(self, mock_plt, sample_data_df):
        """Verifica que plot_distribution guarde archivo cuando se especifica."""
        save_path = Path("/tmp/test_plot.png")

        plots.plot_distribution(sample_data_df, save_path=save_path)

        # Verificar que se llamó savefig
        mock_plt.savefig.assert_called_once_with(save_path, dpi=300, bbox_inches="tight")

    @patch("mlops_project.plots.plt")
    def test_plot_distribution_with_specific_columns(self, mock_plt, sample_data_df):
        """Verifica plot con columnas específicas."""
        columns = ["num1", "num2"]

        plots.plot_distribution(sample_data_df, columns=columns, save_path=None)

        # Debe crear subplots
        mock_plt.subplots.assert_called_once()


class TestPlotBoxplots:
    """Tests para plot_boxplots."""

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.sns")
    def test_plot_boxplots_basic(self, mock_sns, mock_plt, sample_data_df):
        """Verifica que plot_boxplots funcione básicamente."""
        plots.plot_boxplots(sample_data_df, save_path=None)

        # Verificar que se llamó subplots
        mock_plt.subplots.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.sns")
    def test_plot_boxplots_calls_boxplot(self, mock_sns, mock_plt, sample_data_df):
        """Verifica que se llame sns.boxplot."""
        plots.plot_boxplots(sample_data_df, columns=["num1", "num2"], save_path=None)

        # sns.boxplot debe ser llamado para cada columna
        assert mock_sns.boxplot.called


class TestPlotCorrelationMatrix:
    """Tests para plot_correlation_matrix."""

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.sns")
    def test_plot_correlation_matrix_basic(self, mock_sns, mock_plt, sample_data_df):
        """Verifica que plot_correlation_matrix funcione."""
        plots.plot_correlation_matrix(sample_data_df, save_path=None)

        # Verificar que se llamó figure
        mock_plt.figure.assert_called_once()

        # Verificar que se llamó heatmap
        mock_sns.heatmap.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.sns")
    def test_plot_correlation_matrix_only_numeric(self, mock_sns, mock_plt, sample_data_df):
        """Verifica que solo use columnas numéricas."""
        # Agregar columna no numérica
        df_with_cat = sample_data_df.copy()
        df_with_cat["category"] = ["A", "B"] * 50

        plots.plot_correlation_matrix(df_with_cat, save_path=None)

        # Debe seguir funcionando (sklearn maneja esto)
        mock_sns.heatmap.assert_called_once()


class TestPlotConfusionMatrix:
    """Tests para plot_confusion_matrix."""

    @patch("mlops_project.plots.plt")
    def test_plot_confusion_matrix_basic(self, mock_plt, sample_predictions):
        """Verifica que plot_confusion_matrix funcione."""
        y_true, y_pred, _ = sample_predictions

        plots.plot_confusion_matrix(y_true, y_pred, save_path=None)

        # Verificar que se llamó subplots
        mock_plt.subplots.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    def test_plot_confusion_matrix_with_labels(self, mock_plt, sample_predictions):
        """Verifica plot con etiquetas personalizadas."""
        y_true, y_pred, _ = sample_predictions
        labels = ["Malo", "Bueno"]

        plots.plot_confusion_matrix(y_true, y_pred, labels=labels, save_path=None)

        mock_plt.subplots.assert_called_once()


class TestPlotROCCurve:
    """Tests para plot_roc_curve."""

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.RocCurveDisplay")
    def test_plot_roc_curve_basic(self, mock_roc_display, mock_plt, sample_predictions):
        """Verifica que plot_roc_curve funcione."""
        y_true, _, y_proba = sample_predictions

        plots.plot_roc_curve(y_true, y_proba, save_path=None)

        # Verificar que se llamó RocCurveDisplay.from_predictions
        mock_roc_display.from_predictions.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.RocCurveDisplay")
    def test_plot_roc_curve_saves_file(self, mock_roc_display, mock_plt, sample_predictions):
        """Verifica que guarde archivo cuando se especifica."""
        y_true, _, y_proba = sample_predictions
        save_path = Path("/tmp/roc_curve.png")

        plots.plot_roc_curve(y_true, y_proba, save_path=save_path)

        mock_plt.savefig.assert_called_once()


class TestPlotPrecisionRecallCurve:
    """Tests para plot_precision_recall_curve."""

    @patch("mlops_project.plots.plt")
    @patch("mlops_project.plots.PrecisionRecallDisplay")
    def test_plot_pr_curve_basic(self, mock_pr_display, mock_plt, sample_predictions):
        """Verifica que plot_precision_recall_curve funcione."""
        y_true, _, y_proba = sample_predictions

        plots.plot_precision_recall_curve(y_true, y_proba, save_path=None)

        # Verificar que se llamó PrecisionRecallDisplay.from_predictions
        mock_pr_display.from_predictions.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()


class TestPlotFeatureImportance:
    """Tests para plot_feature_importance."""

    @patch("mlops_project.plots.plt")
    def test_plot_feature_importance_basic(self, mock_plt):
        """Verifica que plot_feature_importance funcione."""
        feature_names = [f"feature_{i}" for i in range(30)]
        importances = np.random.rand(30)

        plots.plot_feature_importance(feature_names, importances, top_n=10, save_path=None)

        # Verificar que se llamó figure
        mock_plt.figure.assert_called_once()

        # Verificar que se llamó barh
        mock_plt.barh.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    def test_plot_feature_importance_top_n(self, mock_plt):
        """Verifica que solo muestre top_n features."""
        feature_names = [f"feature_{i}" for i in range(30)]
        importances = np.random.rand(30)
        top_n = 5

        plots.plot_feature_importance(
            feature_names, importances, top_n=top_n, save_path=None
        )

        # Verificar que barh fue llamado con el número correcto de elementos
        call_args = mock_plt.barh.call_args
        assert len(call_args[0][1]) == top_n  # Segundo argumento son los importances


class TestPlotClassDistribution:
    """Tests para plot_class_distribution."""

    @patch("mlops_project.plots.plt")
    def test_plot_class_distribution_with_series(self, mock_plt):
        """Verifica con pd.Series."""
        y = pd.Series([0, 1, 0, 1, 1, 1, 0, 0])

        plots.plot_class_distribution(y, save_path=None)

        # Verificar que se llamó subplots (crea 2 subplots)
        mock_plt.subplots.assert_called_once()

        # Verificar que se llamó show
        mock_plt.show.assert_called_once()

    @patch("mlops_project.plots.plt")
    def test_plot_class_distribution_with_array(self, mock_plt):
        """Verifica con np.array."""
        y = np.array([0, 1, 0, 1, 1, 1, 0, 0])

        plots.plot_class_distribution(y, save_path=None)

        mock_plt.subplots.assert_called_once()

    @patch("mlops_project.plots.plt")
    def test_plot_class_distribution_with_labels(self, mock_plt):
        """Verifica con etiquetas personalizadas."""
        y = np.array([0, 1, 0, 1])
        labels = ["Malo", "Bueno"]

        plots.plot_class_distribution(y, labels=labels, save_path=None)

        mock_plt.subplots.assert_called_once()


class TestCreateReportPlots:
    """Tests para create_report_plots."""

    @patch("mlops_project.plots.plot_class_distribution")
    @patch("mlops_project.plots.plot_precision_recall_curve")
    @patch("mlops_project.plots.plot_roc_curve")
    @patch("mlops_project.plots.plot_confusion_matrix")
    def test_create_report_plots_calls_all_functions(
        self,
        mock_cm,
        mock_roc,
        mock_pr,
        mock_class_dist,
        sample_predictions,
        tmp_path,
    ):
        """Verifica que create_report_plots llame todas las funciones."""
        y_true, y_pred, y_proba = sample_predictions

        plots.create_report_plots(y_true, y_pred, y_proba, save_dir=tmp_path)

        # Verificar que se llamaron todas las funciones de plotting
        mock_cm.assert_called_once()
        mock_roc.assert_called_once()
        mock_pr.assert_called_once()
        mock_class_dist.assert_called_once()

    @patch("mlops_project.plots.plot_class_distribution")
    @patch("mlops_project.plots.plot_precision_recall_curve")
    @patch("mlops_project.plots.plot_roc_curve")
    @patch("mlops_project.plots.plot_confusion_matrix")
    def test_create_report_plots_creates_directory(
        self,
        mock_cm,
        mock_roc,
        mock_pr,
        mock_class_dist,
        sample_predictions,
        tmp_path,
    ):
        """Verifica que cree el directorio si no existe."""
        y_true, y_pred, y_proba = sample_predictions
        save_dir = tmp_path / "new_reports"

        plots.create_report_plots(y_true, y_pred, y_proba, save_dir=save_dir)

        # Verificar que el directorio fue creado
        assert save_dir.exists()

    @patch("mlops_project.plots.plot_class_distribution")
    @patch("mlops_project.plots.plot_precision_recall_curve")
    @patch("mlops_project.plots.plot_roc_curve")
    @patch("mlops_project.plots.plot_confusion_matrix")
    def test_create_report_plots_passes_correct_save_paths(
        self,
        mock_cm,
        mock_roc,
        mock_pr,
        mock_class_dist,
        sample_predictions,
        tmp_path,
    ):
        """Verifica que pase rutas de guardado correctas."""
        y_true, y_pred, y_proba = sample_predictions

        plots.create_report_plots(y_true, y_pred, y_proba, save_dir=tmp_path)

        # Verificar que cada función fue llamada con save_path
        assert mock_cm.call_args[1]["save_path"] is not None
        assert mock_roc.call_args[1]["save_path"] is not None
        assert mock_pr.call_args[1]["save_path"] is not None
        assert mock_class_dist.call_args[1]["save_path"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

