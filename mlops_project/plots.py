"""
Plots module for MLOps project.

Este módulo contiene funciones para crear visualizaciones:
- Gráficos de distribución de datos
- Visualización de outliers
- Curvas ROC y Precision-Recall
- Matrices de confusión
"""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from mlops_project.config import FIGURES_DIR, OUTLIER_VARIABLES


def plot_distribution(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea histogramas para visualizar la distribución de variables.

    Args:
        df: DataFrame con los datos
        columns: Lista de columnas a graficar (si es None, usa todas las numéricas)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()

    n_cols = 4
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        df[col].hist(bins=30, ax=ax, color="#2E86AB", edgecolor="black", alpha=0.7)
        ax.set_title(f"Distribución: {col}", fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Ocultar subplots vacíos
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_boxplots(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea boxplots para visualizar outliers.

    Args:
        df: DataFrame con los datos
        columns: Lista de columnas a graficar (si es None, usa OUTLIER_VARIABLES)
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    if columns is None:
        columns = OUTLIER_VARIABLES

    # Filtrar solo columnas numéricas que existen en el DataFrame
    numeric_cols = df.select_dtypes(include=np.number).columns
    columns = [col for col in columns if col in numeric_cols]

    n_cols = 5
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, col in enumerate(columns):
        ax = axes[idx]
        sns.boxplot(y=df[col], ax=ax, color="#6B8E23")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Ocultar subplots vacíos
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Boxplots para Detección de Outliers ({len(columns)} Variables)",
        fontsize=16,
        fontweight="bold",
        y=1.0,
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea un mapa de calor de la matriz de correlación.

    Args:
        df: DataFrame con los datos
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    # Calcular matriz de correlación solo para variables numéricas
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr()

    # Crear mapa de calor
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )

    plt.title("Matriz de Correlación", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea una visualización de la matriz de confusión.

    Args:
        y_true: Valores verdaderos
        y_pred: Predicciones
        labels: Etiquetas de las clases
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title("Matriz de Confusión", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea la curva ROC.

    Args:
        y_true: Valores verdaderos
        y_proba: Probabilidades de la clase positiva
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    fig, ax = plt.subplots(figsize=figsize)

    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name="Modelo")

    plt.plot([0, 1], [0, 1], "k--", label="Azar", alpha=0.5)
    plt.title("Curva ROC", fontsize=14, fontweight="bold", pad=15)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea la curva Precision-Recall.

    Args:
        y_true: Valores verdaderos
        y_proba: Probabilidades de la clase positiva
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    fig, ax = plt.subplots(figsize=figsize)

    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax, name="Modelo")

    plt.title("Curva Precision-Recall", fontsize=14, fontweight="bold", pad=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea un gráfico de barras con la importancia de features.

    Args:
        feature_names: Nombres de las features
        importances: Valores de importancia
        top_n: Número de top features a mostrar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    # Ordenar por importancia
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], color="#2E86AB", edgecolor="black", alpha=0.8)
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importancia", fontsize=12)
    plt.title(f"Top {top_n} Features Más Importantes", fontsize=14, fontweight="bold", pad=15)
    plt.gca().invert_yaxis()
    plt.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def plot_class_distribution(
    y: pd.Series | np.ndarray,
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Path] = None,
) -> None:
    """
    Crea un gráfico de barras con la distribución de clases.

    Args:
        y: Variable objetivo
        labels: Etiquetas de las clases
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    if isinstance(y, pd.Series):
        counts = y.value_counts().sort_index()
    else:
        unique, counts_array = np.unique(y, return_counts=True)
        counts = pd.Series(counts_array, index=unique)

    # Calcular porcentajes
    percentages = (counts / counts.sum()) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Gráfico de barras
    counts.plot(kind="bar", ax=ax1, color=["#A23B72", "#2E86AB"], edgecolor="black", alpha=0.8)
    ax1.set_title("Distribución de Clases (Conteo)", fontweight="bold")
    ax1.set_xlabel("Clase")
    ax1.set_ylabel("Frecuencia")
    ax1.set_xticklabels(labels if labels else counts.index, rotation=0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Gráfico de pie
    percentages.plot(
        kind="pie",
        ax=ax2,
        autopct="%1.1f%%",
        colors=["#A23B72", "#2E86AB"],
        startangle=90,
        labels=labels if labels else percentages.index,
    )
    ax2.set_title("Distribución de Clases (%)", fontweight="bold")
    ax2.set_ylabel("")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Gráfico guardado en: {save_path}")

    plt.show()


def create_report_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Crea un reporte completo con todos los gráficos de evaluación.

    Args:
        y_true: Valores verdaderos
        y_pred: Predicciones
        y_proba: Probabilidades
        save_dir: Directorio para guardar las figuras
    """
    if save_dir is None:
        save_dir = FIGURES_DIR

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERANDO REPORTE DE VISUALIZACIONES")
    print("=" * 60)

    # 1. Matriz de confusión
    print("\n[1/4] Creando matriz de confusión...")
    plot_confusion_matrix(
        y_true, y_pred, labels=["Malo", "Bueno"], save_path=save_dir / "confusion_matrix.png"
    )

    # 2. Curva ROC
    print("\n[2/4] Creando curva ROC...")
    plot_roc_curve(y_true, y_proba, save_path=save_dir / "roc_curve.png")

    # 3. Curva Precision-Recall
    print("\n[3/4] Creando curva Precision-Recall...")
    plot_precision_recall_curve(y_true, y_proba, save_path=save_dir / "precision_recall_curve.png")

    # 4. Distribución de clases
    print("\n[4/4] Creando distribución de clases...")
    plot_class_distribution(
        y_true, labels=["Malo", "Bueno"], save_path=save_dir / "class_distribution.png"
    )

    print("\n" + "=" * 60)
    print("✓ Reporte de visualizaciones completado")
    print(f"  Gráficos guardados en: {save_dir}")
    print("=" * 60)


# Configurar estilo de matplotlib
sns.set_style("whitegrid")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
