"""
Dataset module for MLOps project.

Este módulo maneja la carga, limpieza inicial y división de datos.
Incluye funcionalidad para:
- Cargar datos desde CSV
- Limpiar la variable objetivo
- Dividir datos en train/test
"""

import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_project.config import (
    MIXED_TYPE_COLUMN,
    RANDOM_SEED,
    STRATIFY,
    TARGET_COLUMN,
    TEST_SIZE,
    get_data_path,
)


class DataLoader:
    """
    Clase para manejar la carga de datos desde archivos CSV.
    """

    def __init__(self, filepath: str | Path, delimiter: str = ","):
        """
        Inicializa el DataLoader.

        Args:
            filepath: Ruta al archivo CSV
            delimiter: Delimitador del CSV (por defecto ',')
        """
        self.filepath = Path(filepath)
        self.delimiter = delimiter
        self.data: Optional[pd.DataFrame] = None

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Carga datos desde el archivo CSV.

        Returns:
            DataFrame con los datos cargados, o None si hay error
        """
        try:
            self.data = pd.read_csv(self.filepath, sep=self.delimiter)
            print(f"✓ Datos cargados exitosamente desde: {self.filepath}")
            print(f"  - Filas: {len(self.data)}")
            print(f"  - Columnas: {len(self.data.columns)}")
            return self.data
        except FileNotFoundError:
            print(f"✗ Error: No se encontró el archivo en: {self.filepath}")
            return None
        except Exception as e:
            print(f"✗ Error al cargar los datos: {e}")
            return None

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Retorna los datos cargados.

        Returns:
            DataFrame con los datos, o None si no se han cargado
        """
        if self.data is None:
            print("⚠ Warning: Los datos no han sido cargados. Llama a load_data() primero.")
        return self.data


class DataCleaner:
    """
    Clase para limpiar datos iniciales:
    - Convertir valores no numéricos a NaN
    - Eliminar columnas mixtas problemáticas
    - Limpiar variable objetivo
    """

    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        mixed_type_column: Optional[str] = MIXED_TYPE_COLUMN,
    ):
        """
        Inicializa el DataCleaner.

        Args:
            target_column: Nombre de la columna objetivo
            mixed_type_column: Columna con tipos mixtos a eliminar
        """
        self.target_column = target_column
        self.mixed_type_column = mixed_type_column

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las operaciones de limpieza al DataFrame.

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame limpio
        """
        print("\n=== Iniciando limpieza de datos ===")

        df_clean = df.copy()

        # 1. Convertir valores no numéricos a NaN
        df_clean = self._convert_to_numeric(df_clean)

        # 2. Eliminar columna de tipo mixto si existe
        if self.mixed_type_column and self.mixed_type_column in df_clean.columns:
            df_clean = df_clean.drop(columns=[self.mixed_type_column])
            print(f"✓ Columna '{self.mixed_type_column}' eliminada")

        # 3. Limpiar variable objetivo
        df_clean = self._clean_target_variable(df_clean)

        print(f"✓ Limpieza completada. Shape final: {df_clean.shape}")
        return df_clean

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas a tipo numérico, reemplazando valores inválidos por NaN.

        Args:
            df: DataFrame a procesar

        Returns:
            DataFrame con conversión numérica aplicada
        """
        df_converted = df.copy()

        # Identificar valores no numéricos para logging
        non_numeric_elements = set()

        for col in df_converted.columns:
            if col == self.mixed_type_column:
                continue

            temp_numeric = pd.to_numeric(df_converted[col], errors="coerce")
            invalid_mask = temp_numeric.isna()

            if invalid_mask.any():
                invalid_values = df_converted[col][invalid_mask].unique()
                non_numeric_elements.update([str(v) for v in invalid_values if pd.notna(v)])

        # Reemplazar valores no numéricos por NaN
        if non_numeric_elements:
            df_converted = df_converted.replace(list(non_numeric_elements), np.nan)
            print(
                f"✓ Valores no numéricos reemplazados por NaN: {len(non_numeric_elements)} tipos únicos"
            )

        # Convertir columnas a numérico
        numeric_cols = [col for col in df_converted.columns if col != self.mixed_type_column]
        df_converted[numeric_cols] = df_converted[numeric_cols].apply(
            lambda x: pd.to_numeric(x, errors="coerce")
        )

        return df_converted

    def _clean_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia la variable objetivo eliminando filas con valores nulos o inválidos.

        Args:
            df: DataFrame a limpiar

        Returns:
            DataFrame sin valores inválidos en la variable objetivo
        """
        if self.target_column not in df.columns:
            print(f"⚠ Warning: Columna objetivo '{self.target_column}' no encontrada")
            return df

        initial_rows = len(df)

        # Mantener solo filas con target válido (0 o 1)
        df_clean = df[
            (~df[self.target_column].isnull()) & (df[self.target_column].isin([0, 1]))
        ].copy()

        removed_rows = initial_rows - len(df_clean)

        print("✓ Variable objetivo limpia:")
        print(f"  - Filas eliminadas: {removed_rows}")
        print(f"  - Filas restantes: {len(df_clean)}")

        return df_clean


class DataSplitter:
    """
    Clase para dividir datos en conjuntos de entrenamiento y prueba.
    """

    def __init__(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_SEED,
        stratify: bool = STRATIFY,
    ):
        """
        Inicializa el DataSplitter.

        Args:
            test_size: Proporción de datos para test (0.0 a 1.0)
            random_state: Semilla aleatoria para reproducibilidad
            stratify: Si se debe estratificar la división por la variable objetivo
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
        self, df: pd.DataFrame, target_column: str = TARGET_COLUMN
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide el DataFrame en conjuntos de entrenamiento y prueba.

        Args:
            df: DataFrame completo
            target_column: Nombre de la columna objetivo

        Returns:
            Tupla (X_train, X_test, y_train, y_test)
        """
        if target_column not in df.columns:
            raise ValueError(f"Columna objetivo '{target_column}' no encontrada en el DataFrame")

        # Separar features y target
        y = df[target_column].copy()
        X = df.drop(columns=[target_column]).copy()

        # Aplicar división
        stratify_param = y if self.stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=stratify_param, random_state=self.random_state
        )

        print(f"\n✓ División de datos completada (test_size={self.test_size}):")
        print(f"  - X_train: {X_train.shape}")
        print(f"  - X_test: {X_test.shape}")
        print(f"  - y_train: {y_train.shape}")
        print(f"  - y_test: {y_test.shape}")

        # Mostrar distribución de clases
        if self.stratify:
            print("\n  Distribución de clases:")
            print(f"    Train: {y_train.value_counts(normalize=True).to_dict()}")
            print(f"    Test:  {y_test.value_counts(normalize=True).to_dict()}")

        return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    filepath: str | Path,
    save_processed: bool = True,
    return_combined: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series] | Tuple[pd.DataFrame, pd.Series]:
    """
    Pipeline completo de carga, limpieza y división de datos.

    Args:
        filepath: Ruta al archivo de datos crudos
        save_processed: Si se debe guardar los datos procesados
        return_combined: Si se debe retornar datos combinados (train+test) en lugar de divididos

    Returns:
        Si return_combined=False: (X_train, X_test, y_train, y_test)
        Si return_combined=True: (X, y)
    """
    print("=" * 60)
    print("PIPELINE DE PREPARACIÓN DE DATOS")
    print("=" * 60)

    # 1. Cargar datos
    loader = DataLoader(filepath)
    df = loader.load_data()

    if df is None:
        raise RuntimeError("No se pudieron cargar los datos")

    # 2. Limpiar datos
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)

    # 3. Guardar datos limpios si se requiere
    if save_processed:
        clean_path = get_data_path("data_clean.csv", "processed")
        df_clean.to_csv(clean_path, index=False)
        print(f"✓ Datos limpios guardados en: {clean_path}")

    # 4. Dividir datos o retornar combinados
    if return_combined:
        # Retornar X, y sin dividir
        y = df_clean[TARGET_COLUMN].copy()
        X = df_clean.drop(columns=[TARGET_COLUMN]).copy()
        print(f"\n✓ Datos preparados (sin división): X{X.shape}, y{y.shape}")
        return X, y
    else:
        # Dividir en train/test
        splitter = DataSplitter()
        X_train, X_test, y_train, y_test = splitter.split(df_clean)

        # Guardar datos divididos si se requiere
        if save_processed:
            # Combinar train y test para guardar
            X_combined = pd.concat([X_train, X_test], axis=0)
            y_combined = pd.concat([y_train, y_test], axis=0)

            X_path = get_data_path("Xtraintest.csv", "processed")
            y_path = get_data_path("ytraintest.csv", "processed")

            X_combined.to_csv(X_path, index=False)
            y_combined.to_csv(y_path, index=False)

            print("\n✓ Datos procesados guardados:")
            print(f"  - Features: {X_path}")
            print(f"  - Target: {y_path}")

        return X_train, X_test, y_train, y_test


def main():
    """
    Función principal para ejecutar el script desde línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Carga y prepara datos para el pipeline de MLOps")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Ruta al archivo CSV de entrada (raw data)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Guardar datos procesados (default: False)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Retornar datos combinados sin dividir (default: False)",
    )

    args = parser.parse_args()

    try:
        result = load_and_prepare_data(
            filepath=args.input,
            save_processed=args.save,
            return_combined=args.combined,
        )

        if args.combined:
            X, y = result
            print("\n✓ Pipeline completado exitosamente")
            print(f"  - Datos combinados: X{X.shape}, y{y.shape}")
        else:
            X_train, X_test, y_train, y_test = result
            print("\n✓ Pipeline completado exitosamente")
            print("  - Datos divididos listos para entrenamiento")

        return 0

    except Exception as e:
        print(f"\n✗ Error en el pipeline: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
