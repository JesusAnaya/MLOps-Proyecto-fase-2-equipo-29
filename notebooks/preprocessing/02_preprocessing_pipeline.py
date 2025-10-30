import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple, Optional

from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import fbeta_score, make_scorer, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


class VariableTypeDefiner:

    def __init__(self):
        self.numericas = ['laufzeit', 'hoehe', 'alter']
        
        self.nominales = [
            'laufkont', 'moral', 'verw', 'sparkont', 'famges',
            'buerge', 'weitkred', 'wohn', 'pers', 'telef', 'gastarb'
        ]
        
        self.ordinales = [
            'beszeit', 'rate', 'wohnzeit', 'verm', 'bishkred', 'beruf'
        ]
    
    def get_all_columns(self) -> list:
        return self.numericas + self.nominales + self.ordinales
    
    def print_summary(self):
        print("\n--- Definición de Tipos de Variables ---")
        print(f"Variables numéricas: {len(self.numericas)}")
        print(f"Variables nominales: {len(self.nominales)}")
        print(f"Variables ordinales: {len(self.ordinales)}")


class PipelineBuilder:

    def __init__(self, variable_types: VariableTypeDefiner):
        self.variable_types = variable_types
        self.preprocessor = None
    
    def build_numeric_pipeline(self) -> Pipeline:
        pipeline = Pipeline(steps=[
            ('impMediana', SimpleImputer(strategy='median')),
            ('escalaNum', MinMaxScaler(feature_range=(1, 2)))
        ])
        return pipeline
    
    def build_nominal_pipeline(self) -> Pipeline:
        pipeline = Pipeline(steps=[
            ('impModa', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        return pipeline
    
    def build_ordinal_pipeline(self) -> Pipeline:
        pipeline = Pipeline(steps=[
            ('impOrd', SimpleImputer(strategy='most_frequent')),
            ('ordtrasnf', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        return pipeline
    
    def build_preprocessor(self) -> ColumnTransformer:
        num_pipeline = self.build_numeric_pipeline()
        nom_pipeline = self.build_nominal_pipeline()
        ord_pipeline = self.build_ordinal_pipeline()
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.variable_types.numericas),
                ('nom', nom_pipeline, self.variable_types.nominales),
                ('ord', ord_pipeline, self.variable_types.ordinales)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        ).set_output(transform="pandas")
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            self.build_preprocessor()
        
        try:
            df_processed = self.preprocessor.fit_transform(df)
            print(" Preprocesamiento completado")
            print(f"  Shape antes: {df.shape}")
            print(f"  Shape después: {df_processed.shape}")
            return df_processed
        except Exception as e:
            print(f"Error durante el preprocesamiento: {e}")
            return None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.preprocessor is None:
            print("El preprocesador no ha sido ajustado (fit) previamente")
            return None
        
        try:
            df_processed = self.preprocessor.transform(df)
            return df_processed
        except Exception as e:
            print(f"Error durante la transformación: {e}")
            return None


class DataSplitter:

    @staticmethod
    def split_by_target(
        df: pd.DataFrame, 
        target_column: str,
        test_size: float = 0.3,
        random_state: int = 1234
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        
        if target_column not in df.columns:
            print(f"ERROR: La columna objetivo '{target_column}' no existe")
            return None, None, None, None
        
        y = df[target_column].copy()
        X = df.drop(columns=[target_column], axis=1).copy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=random_state
        )
        
        print("\n--- División de Datos ---")
        print(f"  División completada")
        print(f"  X Train: {X_train.shape[0]} filas, {X_train.shape[1]} columnas")
        print(f"  X Test: {X_test.shape[0]} filas, {X_test.shape[1]} columnas")
        print(f"  y Train: {y_train.shape[0]} filas")
        print(f"  y Test: {y_test.shape[0]} filas")
        
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def concatenate_splits(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        
        X_combined = pd.concat([X_train, X_test], axis=0)
        y_combined = pd.concat([y_train, y_test], axis=0)
        
        print("\n--- Concatenación de Datos ---")
        print(f" Datos combinados")
        print(f"  Shape X total: {X_combined.shape}")
        print(f"  Shape y total: {y_combined.shape}")
        
        return X_combined, y_combined


class TargetAnalyzer:

    @staticmethod
    def analyze_target_balance(series: pd.Series, target_name: str = 'kredit'):
        print("\n--- Análisis de Balance de la Variable Objetivo ---")
        value_counts = series.value_counts(normalize=True)
        print(value_counts)
        print(f"\nProporción de clases:")
        for value, proportion in value_counts.items():
            print(f"  Clase {value}: {proportion:.2%}")


class DataSaver:

    @staticmethod
    def save_data(data: pd.DataFrame, path: str, filename: str):
        full_path = os.path.join(path, filename)
        try:
            data.to_csv(full_path, index=False)
            print(f" Datos guardados en: {full_path}")
        except Exception as e:
            print(f"Error guardando datos: {e}")
    
    @staticmethod
    def save_preprocessed_data(X: pd.DataFrame, y: pd.Series, base_path: str = "../../data/processed"):
        DataSaver.save_data(X, base_path, "Xtraintest.csv")
        DataSaver.save_data(y.to_frame(), base_path, "ytraintest.csv")


class DataLoader:

    @staticmethod
    def load_cleaned_data(path: str = "../../data/processed/data_clean.csv") -> pd.DataFrame:
        try:
            data = pd.read_csv(path)
            print("  Lectura exitosa de datos limpios")
            print(f"  Filas leídas: {len(data)}")
            print(f"  Columnas: {data.shape[1]}")
            return data
        except FileNotFoundError:
            print(f"ERROR: No se encontró el archivo en: {path}")
            raise
        except Exception as e:
            print(f"ERROR inesperado: {e}")
            raise


class PreprocessingPipeline:

    def __init__(self, data_path: str = "../../data/processed/data_clean.csv"):
        self.data_path = data_path
        self.data = None
        
        self.variable_types = VariableTypeDefiner()
        self.pipeline_builder = PipelineBuilder(self.variable_types)
        self.data_splitter = DataSplitter()
        self.target_analyzer = TargetAnalyzer()
        self.data_saver = DataSaver()
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_combined = None
        self.y_combined = None
        
        self.X_train_processed = None
        self.X_test_processed = None
        self.X_combined_processed = None
    
    def load_data(self) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("PASO 1: Carga de Datos")
        print("=" * 60)
        
        self.data = DataLoader.load_cleaned_data(self.data_path)
        print("\nPrimeras 5 filas:")
        print(self.data.head())
        
        return self.data
    
    def define_variables(self):
        print("\n" + "=" * 60)
        print("PASO 2: Definición de Tipos de Variables")
        print("=" * 60)
        
        self.variable_types.print_summary()
        print("\nVariables numéricas:", self.variable_types.numericas)
        print("\nVariables nominales:", self.variable_types.nominales)
        print("\nVariables ordinales:", self.variable_types.ordinales)
    
    def build_preprocessing_pipelines(self):
        print("\n" + "=" * 60)
        print("PASO 3: Construcción de Pipelines")
        print("=" * 60)
        
        self.pipeline_builder.build_preprocessor()
        print("\n Pipelines construidos:")
        print("  - Pipeline numérico: Imputación (mediana) + Escalado MinMax")
        print("  - Pipeline nominal: Imputación (moda) + One-Hot Encoding")
        print("  - Pipeline ordinal: Imputación (moda) + Ordinal Encoding")
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        print("\n" + "=" * 60)
        print("PASO 4: División de Datos")
        print("=" * 60)
        
        self.target_analyzer.analyze_target_balance(self.data['kredit'])
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.data_splitter.split_by_target(
            self.data, 'kredit'
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def combine_split_data(self):
        print("\n" + "=" * 60)
        print("PASO 5: Combinación de Datos Train/Test")
        print("=" * 60)
        
        self.X_combined, self.y_combined = self.data_splitter.concatenate_splits(
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test
        )
    
    def apply_preprocessing(self):
        print("\n" + "=" * 60)
        print("PASO 6: Aplicación de Transformaciones")
        print("=" * 60)
        
        print("\n--- Procesando datos de entrenamiento ---")
        self.X_train_processed = self.pipeline_builder.fit_transform(self.X_train)
        
        print("\n--- Procesando datos de prueba ---")
        self.X_test_processed = self.pipeline_builder.transform(self.X_test)
        
        print("\n--- Procesando datos combinados ---")
        
        self.pipeline_builder.build_preprocessor()
        self.X_combined_processed = self.pipeline_builder.fit_transform(self.X_combined)
    
    def save_processed_data(self):
        print("\n" + "=" * 60)
        print("PASO 7: Guardado de Datos Procesados")
        print("=" * 60)
        
        self.data_saver.save_preprocessed_data(self.X_combined, self.y_combined)
    
    def run_complete_pipeline(self) -> Tuple[pd.DataFrame, pd.Series]:
        print("\n" + "=" * 60)
        print("PIPELINE DE PREPROCESAMIENTO")
        print("Dataset: German Credit")
        print("=" * 60)
        
        self.load_data()
        self.define_variables()
        self.build_preprocessing_pipelines()
        self.split_data()
        self.combine_split_data()
        self.apply_preprocessing()
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("PREPROCESAMIENTO COMPLETADO EXITOSAMENTE")
        print("=" * 60)
        
        return self.X_combined_processed, self.y_combined


def main():
    
    preprocessor = PreprocessingPipeline()
    X_processed, y = preprocessor.run_complete_pipeline()
    
    print(f"\n Pipeline finalizado")
    print(f"  Dimensiones finales X: {X_processed.shape}")
    print(f"  Dimensiones finales y: {y.shape}")
    
    return preprocessor, X_processed, y


if __name__ == "__main__":
    preprocessor, X_processed, y = main()

