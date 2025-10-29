"""
Training module for MLOps project.

Este módulo maneja el entrenamiento y evaluación de modelos:
- Entrenamiento con cross-validation
- Evaluación de métricas
- Guardado de modelos
- Soporte para balanceo de clases con SMOTE
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional, Tuple
import warnings

from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from mlops_project.config import (
    AVAILABLE_MODELS,
    BEST_MODEL_FILENAME,
    CV_FOLDS,
    CV_REPEATS,
    RANDOM_SEED,
    RESULTS_FILENAME,
    SMOTE_CONFIG,
    get_model_path,
)

# Suprimir warnings durante entrenamiento
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def get_model_instance(model_name: str) -> BaseEstimator:
    """
    Crea una instancia del modelo según el nombre.

    Args:
        model_name: Nombre del modelo ('logistic_regression', 'random_forest', etc.)

    Returns:
        Instancia del modelo configurado

    Raises:
        ValueError: Si el modelo no está disponible
    """
    model_mapping = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "svm": SVC,
        "xgboost": XGBClassifier,
        "mlp": MLPClassifier,
    }

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Modelo '{model_name}' no disponible. Opciones: {list(AVAILABLE_MODELS.keys())}"
        )

    model_class = model_mapping[model_name]
    model_params = AVAILABLE_MODELS[model_name]["params"]

    return model_class(**model_params)


def get_smote_instance(smote_method: str = "SMOTE") -> SMOTE | BorderlineSMOTE:
    """
    Crea una instancia de SMOTE según el método especificado.

    Args:
        smote_method: Método de SMOTE ('SMOTE', 'BorderlineSMOTE')

    Returns:
        Instancia de SMOTE configurada
    """
    if smote_method == "BorderlineSMOTE":
        return BorderlineSMOTE(
            random_state=SMOTE_CONFIG["random_state"],
            k_neighbors=SMOTE_CONFIG["k_neighbors"],
            m_neighbors=SMOTE_CONFIG["m_neighbors"],
        )
    else:
        return SMOTE(random_state=SMOTE_CONFIG["random_state"])


def create_training_pipeline(
    preprocessor: Any,
    model: BaseEstimator,
    use_smote: bool = True,
    smote_method: str = "BorderlineSMOTE",
) -> ImbPipeline:
    """
    Crea el pipeline completo de entrenamiento.

    Args:
        preprocessor: Pipeline de preprocesamiento de features
        model: Modelo a entrenar
        use_smote: Si se debe usar SMOTE para balanceo de clases
        smote_method: Método de SMOTE a usar

    Returns:
        Pipeline de imblearn con preprocesamiento, SMOTE y modelo
    """
    steps = [("preprocessor", preprocessor)]

    if use_smote:
        smote_instance = get_smote_instance(smote_method)
        steps.append(("smote", smote_instance))

    steps.append(("model", model))

    return ImbPipeline(steps=steps)


def evaluate_model(
    pipeline: ImbPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    cv_repeats: int = CV_REPEATS,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Evalúa el modelo usando validación cruzada.

    Args:
        pipeline: Pipeline completo de entrenamiento
        X: Features
        y: Variable objetivo
        cv_folds: Número de folds para cross-validation
        cv_repeats: Número de repeticiones de cross-validation
        verbose: Si se debe imprimir resultados

    Returns:
        Diccionario con métricas de evaluación
    """
    # Configurar cross-validation
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=RANDOM_SEED)

    # Definir métricas
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "geometric_mean": make_scorer(geometric_mean_score),
    }

    # Evaluar con cross-validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_results = cross_validate(
            pipeline, X, np.ravel(y), scoring=scoring, cv=cv, return_train_score=True
        )

    # Calcular estadísticas
    results = {}
    for metric_name in scoring.keys():
        test_key = f"test_{metric_name}"
        train_key = f"train_{metric_name}"

        results[metric_name] = {
            "test_mean": float(np.mean(cv_results[test_key])),
            "test_std": float(np.std(cv_results[test_key])),
            "train_mean": float(np.mean(cv_results[train_key])),
            "train_std": float(np.std(cv_results[train_key])),
        }

    # Imprimir resultados si verbose
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTADOS DE VALIDACIÓN CRUZADA")
        print("=" * 60)
        print(
            f"CV: {cv_folds} folds x {cv_repeats} repeats = {cv_folds * cv_repeats} evaluaciones"
        )
        print("-" * 60)

        for metric_name, scores in results.items():
            print(f"{metric_name.upper():20s}:")
            print(f"  Test:  {scores['test_mean']:.4f} (± {scores['test_std']:.3f})")
            print(f"  Train: {scores['train_mean']:.4f} (± {scores['train_std']:.3f})")

        print("=" * 60)

    return results


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: Any,
    model_name: str = "logistic_regression",
    use_smote: bool = True,
    smote_method: str = "BorderlineSMOTE",
    evaluate: bool = True,
    save_model: bool = True,
    model_filename: Optional[str] = None,
) -> Tuple[ImbPipeline, Optional[Dict[str, Dict[str, float]]]]:
    """
    Entrena un modelo con el pipeline completo.

    Args:
        X_train: Features de entrenamiento
        y_train: Variable objetivo de entrenamiento
        preprocessor: Pipeline de preprocesamiento
        model_name: Nombre del modelo a entrenar
        use_smote: Si se debe usar SMOTE
        smote_method: Método de SMOTE
        evaluate: Si se debe evaluar con cross-validation
        save_model: Si se debe guardar el modelo
        model_filename: Nombre del archivo para guardar el modelo

    Returns:
        Tupla (pipeline_entrenado, resultados_evaluacion)
    """
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO DE MODELO")
    print("=" * 60)

    # Crear instancia del modelo
    model = get_model_instance(model_name)
    model_display_name = AVAILABLE_MODELS[model_name]["name"]

    print(f"\nModelo: {model_display_name}")
    print(f"SMOTE: {'Sí' if use_smote else 'No'} ({smote_method if use_smote else 'N/A'})")
    print(f"Datos: X{X_train.shape}, y{y_train.shape}")

    # Crear pipeline
    pipeline = create_training_pipeline(
        preprocessor=preprocessor,
        model=model,
        use_smote=use_smote,
        smote_method=smote_method,
    )

    # Evaluar con cross-validation si se requiere
    evaluation_results = None
    if evaluate:
        print("\n[1/2] Evaluando modelo con validación cruzada...")
        evaluation_results = evaluate_model(pipeline, X_train, y_train)

    # Entrenar el pipeline completo con todos los datos
    print(
        f"\n[{'2/2' if evaluate else '1/1'}] Entrenando modelo con todos los datos de entrenamiento..."
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    print("✓ Modelo entrenado exitosamente")

    # Guardar modelo si se requiere
    if save_model:
        filename = model_filename or BEST_MODEL_FILENAME
        model_path = get_model_path(filename)
        joblib.dump(pipeline, model_path)
        print(f"✓ Modelo guardado en: {model_path}")

    print("\n" + "=" * 60)

    return pipeline, evaluation_results


def save_results(
    results: Dict[str, Dict[str, float]],
    model_name: str,
    filename: Optional[str] = None,
) -> None:
    """
    Guarda los resultados de evaluación en formato JSON.

    Args:
        results: Resultados de evaluación
        model_name: Nombre del modelo
        filename: Nombre del archivo (opcional)
    """
    filename = filename or RESULTS_FILENAME
    results_path = get_model_path(filename)

    output = {
        "model": model_name,
        "model_display_name": AVAILABLE_MODELS[model_name]["name"],
        "metrics": results,
        "config": {
            "cv_folds": CV_FOLDS,
            "cv_repeats": CV_REPEATS,
            "random_seed": RANDOM_SEED,
            "smote_config": SMOTE_CONFIG,
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"✓ Resultados guardados en: {results_path}")


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de clasificación para el pipeline de MLOps"
    )
    parser.add_argument(
        "--X-train",
        type=str,
        required=True,
        help="Ruta al archivo CSV con features de entrenamiento",
    )
    parser.add_argument(
        "--y-train",
        type=str,
        required=True,
        help="Ruta al archivo CSV con variable objetivo de entrenamiento",
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        required=True,
        help="Ruta al archivo del preprocessor guardado",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Modelo a entrenar",
    )
    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="No usar SMOTE para balanceo de clases",
    )
    parser.add_argument(
        "--smote-method",
        type=str,
        default="BorderlineSMOTE",
        choices=["SMOTE", "BorderlineSMOTE"],
        help="Método de SMOTE a usar",
    )
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="No evaluar con cross-validation",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Nombre del archivo de salida para el modelo (opcional)",
    )

    args = parser.parse_args()

    try:
        # Cargar datos
        print("Cargando datos...")
        X_train = pd.read_csv(args.X_train)
        y_train = pd.read_csv(args.y_train).iloc[:, 0]  # Primera columna

        # Cargar preprocessor
        print(f"Cargando preprocessor desde: {args.preprocessor}")
        preprocessor = joblib.load(args.preprocessor)

        # Entrenar modelo
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name=args.model,
            use_smote=not args.no_smote,
            smote_method=args.smote_method,
            evaluate=not args.no_evaluate,
            save_model=True,
            model_filename=args.output,
        )

        # Guardar resultados si se evaluó
        if results:
            save_results(results, args.model)

        print("\n✓ Pipeline de entrenamiento completado exitosamente")
        return 0

    except Exception as e:
        print(f"\n✗ Error en el pipeline de entrenamiento: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
