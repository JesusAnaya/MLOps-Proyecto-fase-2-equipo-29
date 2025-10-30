import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class DataLoader:

    def __init__(self, base_path: str = "../../data/raw"):

        self.base_path = base_path
    
    def load_dataset(self, filename: str) -> pd.DataFrame:

        filepath = os.path.join(self.base_path, filename)
        
        try:
            data = pd.read_csv(filepath)
            print(f" Lectura exitosa de: {filename}")
            print(f" Filas leídas: {len(data)}")
            return data
        except FileNotFoundError:
            print(f" ERROR: No se encontró el archivo: {filepath}")
            raise
        except Exception as e:
            print(f" ERROR inesperado: {e}")
            raise


class VariableTypeClassifier:

    def __init__(self):
        self.numeric_vars = ['laufzeit', 'hoehe', 'alter']
        self.ordinal_vars = ['beszeit', 'rate', 'wohnzeit', 'verm', 'bishkred', 'beruf']
        self.categorical_vars = [
            'laufkont', 'moral', 'verw', 'sparkont', 'famges', 'buerge',
            'weitkred', 'wohn', 'pers', 'telef', 'gastarb'
        ]
    
    def get_summary(self) -> Dict[str, int]:

        return {
            'numéricas': len(self.numeric_vars),
            'ordinales': len(self.ordinal_vars),
            'categóricas': len(self.categorical_vars)
        }


class InvalidValueDetector:
    
    def __init__(self):
        self.isin_rules = {
            'laufkont': [1, 2, 3, 4],
            'moral': [0, 1, 2, 3, 4],
            'kredit': [0, 1],
            'verw': [2, 0, 9, 3, 1, 10, 5, 4, 6, 8],
            'sparkont': [1, 2, 3, 4, 5],
            'famges': [1, 2, 3, 4],
            'buerge': [1, 2, 3],
            'weitkred': [1, 2, 3],
            'wohn': [1, 2, 3],
            'beszeit': [1, 2, 3, 4, 5],
            'rate': [1, 2, 3, 4],
            'verm': [1, 2, 3, 4],
            'beruf': [1, 2, 3, 4],
            'pers': [1, 2],
            'telef': [1, 2],
            'gastarb': [1, 2]
        }
    
    def detect_non_numeric_values(self, df: pd.DataFrame) -> Dict[str, Dict]:

        non_numeric_summary = {}
        
        for col in df.columns:
            temp_numeric = pd.to_numeric(df[col], errors='coerce')
            non_numeric_elements = temp_numeric.isna()
            
            if non_numeric_elements.any():
                non_numeric_values = df[col][non_numeric_elements]
                non_numeric_summary[col] = non_numeric_values.value_counts().to_dict()
        
        return non_numeric_summary
    
    def replace_non_numeric_with_nan(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

        non_numeric_summary = self.detect_non_numeric_values(df)
        non_numeric_series = pd.Series(non_numeric_summary)
        
        # Extraer elementos no numéricos únicos (excluyendo mixed_type_col)
        all_unique_elements = []
        for count_dict in non_numeric_series[:-1].values:
            all_unique_elements.extend(count_dict.keys())
        
        final_elements = list(set(all_unique_elements))
        
        # Reemplazar con NaN
        df_clean = df.replace(final_elements, np.nan)
        
        return df_clean, final_elements
    
    def convert_to_numeric(self, df: pd.DataFrame, exclude_cols: List[str] = None) -> pd.DataFrame:

        if exclude_cols is None:
            exclude_cols = []
        
        df_clean = df.copy()
        columns_to_convert = [col for col in df_clean.columns if col not in exclude_cols]
        
        for col in columns_to_convert:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        return df_clean
    
    def count_invalid_values(self, df: pd.DataFrame) -> pd.DataFrame:

        results = []
        total_rows = len(df)
        
        for col, valid_values in self.isin_rules.items():
            if col in df.columns:
                mask_valid = df[col].isin(valid_values)
                invalid_count = total_rows - mask_valid.sum()
                
                results.append({
                    'Columna': col,
                    'Regla_Validación': f"isin({valid_values})",
                    'Valores_Invalidos': invalid_count,
                    'Porcentaje_Invalido': np.round((invalid_count / total_rows * 100), 2)
                })
        
        return pd.DataFrame(results)
    
    def filter_by_rules(self, df: pd.DataFrame) -> pd.DataFrame:

        initial_rows = len(df)
        combined_mask = pd.Series(True, index=df.index)
        
        for col, valid_values in self.isin_rules.items():
            if col in df.columns:
                current_mask = df[col].isin(valid_values)
                combined_mask = combined_mask & current_mask
        
        df_cleaned = df[combined_mask].copy()
        removed_rows = initial_rows - len(df_cleaned)
        
        print(f"\n--- Limpieza de Consistencia Categórica ---")
        print(f"Filas originales: {initial_rows}")
        print(f"Filas eliminadas: {removed_rows} ({np.round(100 * removed_rows/initial_rows)}%)")
        print(f"Filas restantes: {len(df_cleaned)}")
        
        return df_cleaned
    
    def impute_with_mode(self, df: pd.DataFrame) -> pd.DataFrame:

        df_imputed = df.copy()
        total_imputations = 0
        
        print("\n--- Iniciando Imputación de Consistencia Categórica (Usando la Moda) ---")
        
        for col, valid_values in self.isin_rules.items():
            if col in df_imputed.columns:
                invalid_mask = ~df_imputed[col].isin(valid_values)
                count_invalid = invalid_mask.sum()
                
                if count_invalid > 0:
                    imputation_mode_value = df_imputed[col].mode()
                    
                    if not imputation_mode_value.empty:
                        mode_value = imputation_mode_value.iloc[0]
                        df_imputed.loc[invalid_mask, col] = mode_value
                        total_imputations += count_invalid
                        print(f"    '{col}': {count_invalid} valores imputados con la moda ({mode_value})")
        
        print(f"Total de valores imputados: {total_imputations}")
        return df_imputed


class TargetCleaner:

    @staticmethod
    def clean_target_nulls(df: pd.DataFrame, target_column: str) -> pd.DataFrame:

        if target_column not in df.columns:
            print(f" ERROR: La columna objetivo '{target_column}' no existe.")
            return df.copy()
        
        initial_rows = len(df)
        df_clean = df[(~df[target_column].isnull()) & (df[target_column].isin([0, 1]))].copy()
        removed_count = initial_rows - len(df_clean)
        
        print("\n--- Limpieza de Nulos en la Variable Objetivo ---")
        print(f"Variable objetivo: '{target_column}'")
        print(f"Registros eliminados: {removed_count}")
        print(f"Filas restantes: {len(df_clean)}")
        
        return df_clean


class OutlierDetector:

    
    def __init__(self, numeric_columns: List[str]):

        self.numeric_columns = numeric_columns
    
    def detect_iqr_outliers(self, df: pd.DataFrame) -> Dict[str, pd.Series]:

        outliers_dict = {}
        
        for column in self.numeric_columns:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
                outliers_dict[column] = outliers
        
        return outliers_dict
    
    def plot_boxplots(self, df: pd.DataFrame, save_path: Optional[str] = None):
 
        df_numeric = df.select_dtypes(include=np.number)
        variables = [var for var in self.numeric_columns if var in df_numeric.columns]
        
        num_vars = len(variables)
        n_cols = 5
        n_rows = (num_vars + n_cols - 1) // n_cols
        
        plt.figure(figsize=(4 * n_cols, 4 * n_rows))
        
        for i, var in enumerate(variables):
            ax = plt.subplot(n_rows, n_cols, i + 1)
            sns.boxplot(y=df_numeric[var], ax=ax, color='#6B8E23')
            ax.set_title(var, fontsize=12)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(axis='y', labelsize=10)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.suptitle(f"Boxplots para {num_vars} Variables Numéricas", fontsize=20, y=1.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_outlier_summary(self, df: pd.DataFrame):

        outliers_dict = self.detect_iqr_outliers(df)
        n = len(df)
        outlier_count = 0
        
        print("\n--- DETECCIÓN DE OUTLIERS (Método IQR) ---")
        for var, outliers in outliers_dict.items():
            if not outliers.empty:
                print(f"[{var}]: {len(outliers)} outliers encontrados, "
                      f"{np.round(100 * len(outliers)/n, 2)}% del total")
                outlier_count += len(outliers)
        
        if outlier_count == 0:
            print("No se detectaron outliers en las variables seleccionadas.")


class GermanCreditEDA:

    def __init__(self, data_path: str = "../../data/raw"):

        self.data_path = data_path
        self.loader = DataLoader(data_path)
        self.variable_classifier = VariableTypeClassifier()
        self.invalid_detector = InvalidValueDetector()
        self.target_cleaner = TargetCleaner()
        
        # Columnas numéricas para detección de outliers
        numeric_cols = ['laufzeit', 'wohnzeit', 'alter', 'bishkred', 'hoehe']
        self.outlier_detector = OutlierDetector(numeric_cols)
        
        # Almacenamiento de datos
        self.original_data = None
        self.modified_data = None
        self.cleaned_data = None
        self.imputed_data = None
    
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        print("=" * 60)
        print("PASO 1: Carga de Datos")
        print("=" * 60)
        
        self.original_data = self.loader.load_dataset("german_credit_original.csv")
        self.modified_data = self.loader.load_dataset("german_credit_modified.csv")
        
        print("\nPrimeras 5 filas de datos originales:")
        print(self.original_data.head())
        print("\nPrimeras 5 filas de datos modificados:")
        print(self.modified_data.head())
        
        return self.original_data, self.modified_data
    
    def detect_and_clean_non_numeric(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\n" + "=" * 60)
        print("PASO 2: Detección y Limpieza de Valores No Numéricos")
        print("=" * 60)
        
        # Detectar valores no numéricos
        non_numeric_summary = self.invalid_detector.detect_non_numeric_values(df)
        print("\nValores no numéricos detectados:")
        print(pd.Series(non_numeric_summary))
        
        # Reemplazar con NaN
        df_clean, replaced_elements = self.invalid_detector.replace_non_numeric_with_nan(df)
        print(f"\nElementos reemplazados por NaN: {replaced_elements}")
        print(f"\nConteo de nulos después de la limpieza:")
        print(df_clean.isnull().sum())
        
        # Convertir a numérico
        df_numeric = self.invalid_detector.convert_to_numeric(
            df_clean, 
            exclude_cols=['mixed_type_col']
        )
        
        print(f"\nDimensiones del dataset limpio: {df_numeric.shape}")
        print("\nEstadísticas descriptivas:")
        print(df_numeric.describe())
        
        return df_numeric
    
    def clean_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\n" + "=" * 60)
        print("PASO 3: Limpieza de Variable Objetivo")
        print("=" * 60)
        
        df_clean = self.target_cleaner.clean_target_nulls(df, 'kredit')
        
        # Eliminar mixed_type_col
        if 'mixed_type_col' in df_clean.columns:
            df_clean = df_clean.drop('mixed_type_col', axis=1)
            print(" Columna 'mixed_type_col' eliminada")
        
        return df_clean
    
    def handle_invalid_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\n" + "=" * 60)
        print("PASO 4: Imputación de Valores No Válidos en Variables Categóricas")
        print("=" * 60)
        
        # Contar valores no válidos
        invalid_counts = self.invalid_detector.count_invalid_values(df)
        print("\nConteo de valores no válidos por columna:")
        print(invalid_counts)
        print(f"\nTotal de valores inválidos: {invalid_counts['Valores_Invalidos'].sum()}")
        
        # Imputar con la moda
        df_imputed = self.invalid_detector.impute_with_mode(df)
        
        print("\nEstadísticas descriptivas después de la imputación:")
        print(df_imputed.describe())
        
        return df_imputed
    
    def detect_outliers(self, df: pd.DataFrame, plot: bool = True):

        print("\n" + "=" * 60)
        print("PASO 5: Detección de Outliers")
        print("=" * 60)
        
        self.outlier_detector.print_outlier_summary(df)
        
        if plot:
            print("\n--- Generando Gráficos de Boxplots ---")
            self.outlier_detector.plot_boxplots(df)
    
    def save_cleaned_data(self, output_path: str):

        if self.cleaned_data is not None:
            self.cleaned_data.to_csv(output_path, index=False)
            print(f"\n Datos limpios guardados en: {output_path}")
        else:
            print("\n No hay datos limpios para guardar")
    
    def run_complete_analysis(self) -> pd.DataFrame:

        print("\n" + "=" * 60)
        print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
        print("Dataset: German Credit")
        print("=" * 60)
        
        # Paso 1: Cargar datos
        self.load_datasets()
        
        # Paso 2: Limpiar valores no numéricos
        self.cleaned_data = self.detect_and_clean_non_numeric(self.modified_data)
        
        # Paso 3: Limpiar variable objetivo
        self.cleaned_data = self.clean_target_variable(self.cleaned_data)
        
        # Paso 4: Manejar valores no válidos en variables categóricas
        self.imputed_data = self.handle_invalid_categorical_values(self.cleaned_data)
        
        # Paso 5: Detectar outliers
        self.detect_outliers(self.imputed_data, plot=True)
        
        # Guardar datos limpios
        output_path = os.path.join('..', '..', 'data', 'processed', 'data_clean.csv')
        self.save_cleaned_data(output_path)
        
        print("\n" + "=" * 60)
        print("EDA COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return self.imputed_data


def main():

    # Inicializar EDA analyzer
    eda = GermanCreditEDA()
    
    # Ejecutar análisis completo
    final_data = eda.run_complete_analysis()
    
    print(f"\nDimensiones finales del dataset: {final_data.shape}")
    print(f"Total de registros: {len(final_data)}")
    
    return eda, final_data


if __name__ == "__main__":
    eda, final_data = main()

