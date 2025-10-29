"""
Features module for MLOps project.

Este módulo maneja la ingeniería de características:
- Validación de datos categóricos
- Imputación de valores inválidos
- Transformación de features (escalado, encoding)
- Manejo de outliers
"""

import argparse
import sys
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from mlops_project.config import (
    CATEGORICAL_IMPUTE_STRATEGY,
    CATEGORICAL_VALIDATION_RULES,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    NUMERIC_IMPUTE_STRATEGY,
    NUMERIC_SCALER_RANGE,
    ORDINAL_FEATURES,
    OUTLIER_METHOD,
    OUTLIER_PERCENTILES,
    OUTLIER_VARIABLES,
    get_model_path,
)


class InvalidDataHandler(BaseEstimator, TransformerMixin):
    """
    Clase para manejar valores inválidos en variables categóricas.
    Imputa valores no válidos usando la moda.
    """

    def __init__(self, validation_rules: Optional[Dict[str, List[int]]] = None):
        """
        Inicializa el InvalidDataHandler.

        Args:
            validation_rules: Diccionario con reglas de validación {columna: [valores_válidos]}
        """
        self.validation_rules = validation_rules or CATEGORICAL_VALIDATION_RULES
        self.mode_values_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Aprende la moda de cada columna categórica.

        Args:
            X: DataFrame de entrenamiento
            y: Variable objetivo (no usada, solo para compatibilidad con sklearn)

        Returns:
            self
        """
        X_copy = X.copy()

        for col, valid_values in self.validation_rules.items():
            if col in X_copy.columns:
                # Calcular moda para valores válidos
                valid_mask = X_copy[col].isin(valid_values)
                valid_data = X_copy.loc[valid_mask, col]

                if len(valid_data) > 0:
                    mode_value = valid_data.mode()
                    if not mode_value.empty:
                        self.mode_values_[col] = mode_value.iloc[0]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Imputa valores inválidos con la moda aprendida.

        Args:
            X: DataFrame a transformar

        Returns:
            DataFrame con valores inválidos imputados
        """
        X_imputed = X.copy()
        total_imputations = 0

        for col, valid_values in self.validation_rules.items():
            if col not in X_imputed.columns:
                continue

            # Identificar valores inválidos
            invalid_mask = ~X_imputed[col].isin(valid_values)
            count_invalid = invalid_mask.sum()

            if count_invalid > 0 and col in self.mode_values_:
                # Imputar con la moda
                X_imputed.loc[invalid_mask, col] = self.mode_values_[col]
                total_imputations += count_invalid

        if total_imputations > 0:
            print(f"✓ Valores inválidos imputados: {total_imputations}")

        return X_imputed


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Clase para detectar y manejar outliers en variables numéricas.
    Usa método IQR o Percentiles para delimitar valores extremos.
    """

    def __init__(
        self,
        method: str = OUTLIER_METHOD,
        percentiles: Tuple[float, float] = OUTLIER_PERCENTILES,
        variables: Optional[List[str]] = None,
    ):
        """
        Inicializa el OutlierHandler.

        Args:
            method: Método de detección ('IQR' o 'Percentiles')
            percentiles: Tupla (p_low, p_high) para método Percentiles
            variables: Lista de variables numéricas a procesar
        """
        if method not in ["IQR", "Percentiles"]:
            raise ValueError("method debe ser 'IQR' o 'Percentiles'")

        self.method = method
        self.percentiles = percentiles
        self.variables = variables or OUTLIER_VARIABLES
        self.limits_: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Calcula los límites de outliers para cada variable.

        Args:
            X: DataFrame de entrenamiento
            y: Variable objetivo (no usada)

        Returns:
            self
        """
        X_numeric = X[self.variables].select_dtypes(include=np.number)

        for var in self.variables:
            if var not in X_numeric.columns:
                continue

            if self.method == "IQR":
                Q1 = X_numeric[var].quantile(0.25)
                Q3 = X_numeric[var].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

            else:  # Percentiles
                p_low, p_high = self.percentiles
                lower_bound = X_numeric[var].quantile(p_low)
                upper_bound = X_numeric[var].quantile(p_high)

            self.limits_[var] = {"lower": lower_bound, "upper": upper_bound}

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica capping a los outliers.

        Args:
            X: DataFrame a transformar

        Returns:
            DataFrame con outliers delimitados
        """
        X_capped = X.copy()
        total_caps = 0

        for var, limits in self.limits_.items():
            if var not in X_capped.columns:
                continue

            lower = limits["lower"]
            upper = limits["upper"]

            # Identificar outliers
            is_lower_outlier = X_capped[var] < lower
            is_upper_outlier = X_capped[var] > upper

            count_caps = is_lower_outlier.sum() + is_upper_outlier.sum()
            total_caps += count_caps

            if count_caps > 0:
                # Aplicar capping
                X_capped.loc[is_lower_outlier, var] = lower
                X_capped.loc[is_upper_outlier, var] = upper

        if total_caps > 0:
            print(f"✓ Outliers delimitados: {total_caps} valores")

        return X_capped


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """
    Clase principal para preprocesar features.
    Combina imputación, escalado y encoding usando ColumnTransformer.
    """

    def __init__(
        self,
        numeric_features: Optional[List[str]] = None,
        nominal_features: Optional[List[str]] = None,
        ordinal_features: Optional[List[str]] = None,
        numeric_impute_strategy: str = NUMERIC_IMPUTE_STRATEGY,
        categorical_impute_strategy: str = CATEGORICAL_IMPUTE_STRATEGY,
        scaler_range: Tuple[int, int] = NUMERIC_SCALER_RANGE,
    ):
        """
        Inicializa el FeaturePreprocessor.

        Args:
            numeric_features: Lista de features numéricas
            nominal_features: Lista de features categóricas nominales
            ordinal_features: Lista de features categóricas ordinales
            numeric_impute_strategy: Estrategia de imputación para numéricas
            categorical_impute_strategy: Estrategia de imputación para categóricas
            scaler_range: Rango del MinMaxScaler
        """
        self.numeric_features = numeric_features or NUMERIC_FEATURES
        self.nominal_features = nominal_features or NOMINAL_FEATURES
        self.ordinal_features = ordinal_features or ORDINAL_FEATURES
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.scaler_range = scaler_range
        self.preprocessor_: Optional[ColumnTransformer] = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Construye y ajusta el ColumnTransformer.

        Args:
            X: DataFrame de entrenamiento
            y: Variable objetivo (no usada)

        Returns:
            self
        """
        # Pipeline para variables numéricas
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.numeric_impute_strategy)),
                ("scaler", MinMaxScaler(feature_range=self.scaler_range)),
            ]
        )

        # Pipeline para variables nominales
        nominal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # Pipeline para variables ordinales
        ordinal_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.categorical_impute_strategy)),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )

        # Combinar pipelines
        self.preprocessor_ = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("nom", nominal_pipeline, self.nominal_features),
                ("ord", ordinal_pipeline, self.ordinal_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        # Ajustar el preprocessor
        self.preprocessor_.fit(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones al DataFrame.

        Args:
            X: DataFrame a transformar

        Returns:
            DataFrame transformado
        """
        if self.preprocessor_ is None:
            raise RuntimeError("El preprocessor no ha sido ajustado. Llama a fit() primero.")

        X_transformed = self.preprocessor_.transform(X)

        print(f"✓ Features transformadas: {X.shape} -> {X_transformed.shape}")

        return X_transformed

    def get_feature_names_out(self) -> List[str]:
        """
        Obtiene los nombres de las features después de la transformación.

        Returns:
            Lista de nombres de features
        """
        if self.preprocessor_ is None:
            raise RuntimeError("El preprocessor no ha sido ajustado. Llama a fit() primero.")

        return list(self.preprocessor_.get_feature_names_out())


def create_feature_pipeline(
    include_invalid_handler: bool = True,
    include_outlier_handler: bool = True,
) -> Pipeline:
    """
    Crea el pipeline completo de preprocesamiento de features.

    Args:
        include_invalid_handler: Si se debe incluir manejo de valores inválidos
        include_outlier_handler: Si se debe incluir manejo de outliers

    Returns:
        Pipeline de scikit-learn con todos los pasos de preprocesamiento
    """
    steps = []

    # 1. Manejo de valores inválidos (opcional)
    if include_invalid_handler:
        steps.append(("invalid_handler", InvalidDataHandler()))

    # 2. Manejo de outliers (opcional)
    if include_outlier_handler:
        steps.append(("outlier_handler", OutlierHandler()))

    # 3. Preprocesamiento principal (siempre incluido)
    steps.append(("feature_preprocessor", FeaturePreprocessor()))

    pipeline = Pipeline(steps=steps)

    return pipeline


def prepare_features(
    X_train: pd.DataFrame,
    X_test: Optional[pd.DataFrame] = None,
    save_preprocessor: bool = True,
    preprocessor_name: str = "preprocessor.joblib",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Pipeline]:
    """
    Prepara features usando el pipeline completo.

    Args:
        X_train: DataFrame de entrenamiento
        X_test: DataFrame de prueba (opcional)
        save_preprocessor: Si se debe guardar el preprocessor ajustado
        preprocessor_name: Nombre del archivo para guardar el preprocessor

    Returns:
        Tupla (X_train_transformed, X_test_transformed, pipeline)
    """
    print("\n" + "=" * 60)
    print("PIPELINE DE PREPARACIÓN DE FEATURES")
    print("=" * 60)

    # Crear pipeline
    pipeline = create_feature_pipeline()

    # Ajustar con datos de entrenamiento
    print("\n[1/3] Ajustando pipeline con datos de entrenamiento...")
    pipeline.fit(X_train)

    # Transformar datos de entrenamiento
    print("\n[2/3] Transformando datos de entrenamiento...")
    X_train_transformed = pipeline.transform(X_train)

    # Transformar datos de prueba si existen
    X_test_transformed = None
    if X_test is not None:
        print("\n[3/3] Transformando datos de prueba...")
        X_test_transformed = pipeline.transform(X_test)

    # Guardar preprocessor si se requiere
    if save_preprocessor:
        preprocessor_path = get_model_path(preprocessor_name)
        joblib.dump(pipeline, preprocessor_path)
        print(f"\n✓ Preprocessor guardado en: {preprocessor_path}")

    print("\n" + "=" * 60)
    print("✓ Preparación de features completada")
    print("=" * 60)

    return X_train_transformed, X_test_transformed, pipeline


def load_preprocessor(preprocessor_name: str = "preprocessor.joblib") -> Pipeline:
    """
    Carga un preprocessor previamente guardado.

    Args:
        preprocessor_name: Nombre del archivo del preprocessor

    Returns:
        Pipeline cargado
    """
    preprocessor_path = get_model_path(preprocessor_name)

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor no encontrado en: {preprocessor_path}")

    pipeline = joblib.load(preprocessor_path)
    print(f"✓ Preprocessor cargado desde: {preprocessor_path}")

    return pipeline


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Prepara features para el pipeline de MLOps")
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Ruta al archivo CSV con datos de entrenamiento",
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Ruta al archivo CSV con datos de prueba (opcional)",
    )
    parser.add_argument(
        "--output-train",
        type=str,
        default="X_train_processed.csv",
        help="Ruta de salida para datos de entrenamiento procesados",
    )
    parser.add_argument(
        "--output-test",
        type=str,
        default="X_test_processed.csv",
        help="Ruta de salida para datos de prueba procesados",
    )
    parser.add_argument(
        "--save-preprocessor",
        action="store_true",
        help="Guardar el preprocessor ajustado",
    )

    args = parser.parse_args()

    try:
        # Cargar datos
        X_train = pd.read_csv(args.train)
        X_test = pd.read_csv(args.test) if args.test else None

        # Preparar features
        X_train_transformed, X_test_transformed, pipeline = prepare_features(
            X_train=X_train,
            X_test=X_test,
            save_preprocessor=args.save_preprocessor,
        )

        # Guardar datos transformados
        X_train_transformed.to_csv(args.output_train, index=False)
        print(f"✓ Datos de entrenamiento guardados en: {args.output_train}")

        if X_test_transformed is not None:
            X_test_transformed.to_csv(args.output_test, index=False)
            print(f"✓ Datos de prueba guardados en: {args.output_test}")

        print("\n✓ Pipeline completado exitosamente")
        return 0

    except Exception as e:
        print(f"\n✗ Error en el pipeline: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
