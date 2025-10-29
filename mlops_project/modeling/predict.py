"""
Prediction module for MLOps project.

Este módulo maneja la inferencia con modelos entrenados:
- Cargar modelos guardados
- Realizar predicciones
- Calcular probabilidades
- Evaluar en datos de test
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from mlops_project.config import BEST_MODEL_FILENAME, get_model_path


def load_model(model_filename: str = BEST_MODEL_FILENAME):
    """
    Carga un modelo previamente guardado.

    Args:
        model_filename: Nombre del archivo del modelo

    Returns:
        Pipeline del modelo cargado

    Raises:
        FileNotFoundError: Si el modelo no existe
    """
    model_path = get_model_path(model_filename)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

    model = joblib.load(model_path)
    print(f"✓ Modelo cargado desde: {model_path}")

    return model


def predict(
    model,
    X: pd.DataFrame,
    return_proba: bool = False,
) -> np.ndarray:
    """
    Realiza predicciones con el modelo.

    Args:
        model: Modelo o pipeline entrenado
        X: Features para predicción
        return_proba: Si se deben retornar probabilidades en lugar de clases

    Returns:
        Array con predicciones (clases o probabilidades)
    """
    if return_proba:
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X)
            # Retornar solo probabilidad de clase positiva (índice 1)
            return predictions[:, 1]
        else:
            raise AttributeError("El modelo no soporta predict_proba")
    else:
        return model.predict(X)


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evalúa las predicciones calculando métricas.

    Args:
        y_true: Valores verdaderos
        y_pred: Predicciones (clases)
        y_proba: Probabilidades de clase positiva (opcional)
        verbose: Si se debe imprimir resultados

    Returns:
        Diccionario con métricas de evaluación
    """
    # Calcular métricas básicas
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # Calcular métricas que requieren probabilidades
    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["average_precision"] = float(average_precision_score(y_true, y_proba))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }

    # Imprimir resultados si verbose
    if verbose:
        print("\n" + "=" * 60)
        print("MÉTRICAS DE EVALUACIÓN")
        print("=" * 60)
        print(f"Accuracy:          {metrics['accuracy']:.4f}")
        print(f"Precision:         {metrics['precision']:.4f}")
        print(f"Recall:            {metrics['recall']:.4f}")
        print(f"F1-Score:          {metrics['f1']:.4f}")

        if y_proba is not None:
            print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
            print(f"Average Precision: {metrics['average_precision']:.4f}")

        print("-" * 60)
        print("MATRIZ DE CONFUSIÓN")
        print("-" * 60)
        print(
            f"TN: {metrics['confusion_matrix']['true_negatives']:4d}  |  FP: {metrics['confusion_matrix']['false_positives']:4d}"
        )
        print(
            f"FN: {metrics['confusion_matrix']['false_negatives']:4d}  |  TP: {metrics['confusion_matrix']['true_positives']:4d}"
        )
        print("=" * 60)

    return metrics


def predict_and_evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_predictions: bool = False,
    output_file: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Pipeline completo de predicción y evaluación.

    Args:
        model: Modelo entrenado
        X_test: Features de test
        y_test: Variable objetivo de test
        save_predictions: Si se deben guardar las predicciones
        output_file: Archivo para guardar predicciones

    Returns:
        Tupla (predicciones, probabilidades, métricas)
    """
    print("\n" + "=" * 60)
    print("PREDICCIÓN Y EVALUACIÓN")
    print("=" * 60)

    # Realizar predicciones
    print(f"\nRealizando predicciones en {len(X_test)} muestras...")
    y_pred = predict(model, X_test, return_proba=False)

    # Obtener probabilidades si el modelo lo soporta
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = predict(model, X_test, return_proba=True)

    # Evaluar predicciones
    metrics = evaluate_predictions(y_test, y_pred, y_proba, verbose=True)

    # Guardar predicciones si se requiere
    if save_predictions:
        predictions_df = pd.DataFrame(
            {
                "y_true": y_test,
                "y_pred": y_pred,
            }
        )

        if y_proba is not None:
            predictions_df["y_proba"] = y_proba

        output_path = output_file or "predictions.csv"
        predictions_df.to_csv(output_path, index=False)
        print(f"\n✓ Predicciones guardadas en: {output_path}")

    return y_pred, y_proba, metrics


def batch_predict(
    model,
    X: pd.DataFrame,
    batch_size: int = 1000,
    return_proba: bool = False,
) -> np.ndarray:
    """
    Realiza predicciones en batches (útil para datasets grandes).

    Args:
        model: Modelo entrenado
        X: Features para predicción
        batch_size: Tamaño del batch
        return_proba: Si se deben retornar probabilidades

    Returns:
        Array con todas las predicciones
    """
    n_samples = len(X)
    n_batches = (n_samples + batch_size - 1) // batch_size

    predictions = []

    print(f"Realizando predicciones en {n_batches} batches de {batch_size}...")

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)

        X_batch = X.iloc[start_idx:end_idx]
        batch_pred = predict(model, X_batch, return_proba=return_proba)

        predictions.append(batch_pred)

        if (i + 1) % 10 == 0:
            print(f"  Procesados: {end_idx}/{n_samples} muestras")

    return np.concatenate(predictions)


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Realiza predicciones con un modelo entrenado")
    parser.add_argument(
        "--model",
        type=str,
        default=BEST_MODEL_FILENAME,
        help="Ruta al archivo del modelo",
    )
    parser.add_argument(
        "--X-test",
        type=str,
        required=True,
        help="Ruta al archivo CSV con features de test",
    )
    parser.add_argument(
        "--y-test",
        type=str,
        help="Ruta al archivo CSV con variable objetivo de test (opcional para evaluación)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Archivo de salida para predicciones",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Tamaño del batch para predicciones",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Guardar probabilidades en lugar de clases",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar predicciones en archivo",
    )

    args = parser.parse_args()

    try:
        # Cargar modelo
        model = load_model(args.model)

        # Cargar datos de test
        print(f"Cargando datos de test desde: {args.X_test}")
        X_test = pd.read_csv(args.X_test)

        # Si hay variable objetivo, evaluar
        if args.y_test:
            print(f"Cargando variable objetivo desde: {args.y_test}")
            y_test = pd.read_csv(args.y_test).iloc[:, 0]

            # Predicción y evaluación completa
            y_pred, y_proba, metrics = predict_and_evaluate(
                model=model,
                X_test=X_test,
                y_test=y_test,
                save_predictions=args.save,
                output_file=args.output,
            )

            # Guardar métricas
            metrics_path = Path(args.output).parent / "evaluation_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"\n✓ Métricas guardadas en: {metrics_path}")

        else:
            # Solo predicción sin evaluación
            print("\nRealizando predicciones...")

            if len(X_test) > args.batch_size:
                predictions = batch_predict(
                    model, X_test, batch_size=args.batch_size, return_proba=args.proba
                )
            else:
                predictions = predict(model, X_test, return_proba=args.proba)

            print(f"✓ Predicciones completadas: {len(predictions)} muestras")

            # Guardar predicciones si se requiere
            if args.save:
                column_name = "y_proba" if args.proba else "y_pred"
                predictions_df = pd.DataFrame({column_name: predictions})
                predictions_df.to_csv(args.output, index=False)
                print(f"✓ Predicciones guardadas en: {args.output}")

        print("\n✓ Pipeline de predicción completado exitosamente")
        return 0

    except Exception as e:
        print(f"\n✗ Error en el pipeline de predicción: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
