import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
import os
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.metrics import geometric_mean_score
import shap


class DataLoader:
    def __init__(self, base_path: str = "../../data/raw/"):
        self.base_path = Path(base_path)
    
    def load_dataset(self, filename: str, description: str = "") -> pd.DataFrame:
        file_path = self.base_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"El archivo '{filename}' no se encontró en la ruta: {self.base_path}"
            )
        
        print(f"\n=== CARGANDO DATASET: {filename} ===")
        if description:
            print(f"Descripción: {description}")
        
        df = pd.read_csv(file_path)
        
        print(f" Dataset cargado exitosamente")
        print(f" Dimensiones: {df.shape[0]} filas, {df.shape[1]} columnas")
        print(f" Ruta: {file_path}")
        
        return df
    
    def load_modified_dataset(self) -> pd.DataFrame:
        return self.load_dataset(
            filename="german_credit_modified.csv",
            description="Dataset alemán de créditos modificado (con valores inválidos para análisis)"
        )
    
    def load_original_dataset(self) -> pd.DataFrame:
        return self.load_dataset(
            filename="german_credit_original.csv",
            description="Dataset alemán de créditos original (sin modificaciones)"
        )
    
    def compare_datasets(self, df_original: pd.DataFrame, df_modified: pd.DataFrame) -> None:
        print("\n=== COMPARACIÓN DE DATASETS ===")
        print(f"Dataset Original:  {df_original.shape[0]} filas, {df_original.shape[1]} columnas")
        print(f"Dataset Modificado: {df_modified.shape[0]} filas, {df_modified.shape[1]} columnas")
        print(f"\nDiferencias:")
        print(f"  - Filas: {df_modified.shape[0] - df_original.shape[0]} ({df_modified.shape[0] - df_original.shape[0]})")
        print(f"  - Columnas: {df_modified.shape[1] - df_original.shape[1]} ({df_modified.shape[1] - df_original.shape[1]})")
        
        if df_modified.shape[1] > df_original.shape[1]:
            extra_cols = set(df_modified.columns) - set(df_original.columns)
            print(f"\nColumnas adicionales en el dataset modificado: {extra_cols}")


class DataConfiguration:
    
    def __init__(self):
        self.numerical_vars = ['laufzeit', 'hoehe', 'alter']
        self.ordinal_vars = ['beszeit', 'rate', 'wohnzeit', 'verm', 'bishkred', 'beruf']
        self.categorical_vars = [
            'laufkont', 'moral', 'verw', 'sparkont', 'famges', 'buerge',
            'weitkred', 'wohn', 'pers', 'telef', 'gastarb'
        ]
        
        self.validation_rules = {
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
            'gastarb': [1, 2],
            'kredit': [0, 1]
        }
    
    def get_all_features(self):

        return self.numerical_vars + self.ordinal_vars + self.categorical_vars


class DataCleaner:

    def __init__(self, config: DataConfiguration):
        self.config = config
    
    def clean_invalid_numeric_values(self, df: pd.DataFrame) -> pd.DataFrame:

        non_numeric_summary = {}
        
        for col in df.columns:
            temp_numeric = pd.to_numeric(df[col], errors='coerce')
            non_numeric_elements = temp_numeric.isna()
            
            if non_numeric_elements.any():
                non_numeric_values = df[col][non_numeric_elements]
                non_numeric_summary[col] = non_numeric_values.value_counts().to_dict()
        
        all_unique_non_numeric_elements = []
        for count_dict in list(non_numeric_summary.values()):  
            all_unique_non_numeric_elements.extend(count_dict.keys())

        final_elements = list(set(all_unique_non_numeric_elements))
        
        df_clean = df.replace(final_elements, np.nan)
        
        print("Verificación de conteo de nulos:")
        print(df_clean.isnull().sum())
        
        columnas_numericas = df_clean.columns[df_clean.columns != 'mixed_type_col'].tolist()
        df_clean[columnas_numericas] = df_clean[columnas_numericas].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        )
        
        print(f"Shape después de limpieza: {df_clean.shape}")
        
        return df_clean
    
    def clean_target_nulls(self, df: pd.DataFrame, target_column_name: str = 'kredit') -> pd.DataFrame:

        if target_column_name not in df.columns:
            print(f"Error: La columna objetivo '{target_column_name}' no existe en el DataFrame.")
            return df.copy()
        
        initial_rows = len(df)
        
        df_clean_target = df[
            (~df[target_column_name].isnull()) & 
            (df[target_column_name].isin([0, 1]))
        ].copy()
        
        removed_count = initial_rows - len(df_clean_target)
        
        print("--- Limpieza de Nulos en la Variable Objetivo ---")
        print(f"Variable objetivo: '{target_column_name}'")
        print(f"Registros eliminados: {removed_count}")
        print(f"Filas restantes: {len(df_clean_target)}")
        print("-" * 50)
        
        return df_clean_target
    
    def validate_categorical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:

        print('--- Conteos de valores no válidos ---')
        results = []
        total_rows = len(df)
        
        for col, valid_values in self.config.validation_rules.items():
            if col in df.columns:
                mask_valid = df[col].isin(valid_values)
                invalid_count = total_rows - mask_valid.sum()
                
                results.append({
                    'Columna': col,
                    'Regla_Validación': f"isin({valid_values})",
                    'Valores_Invalidos': invalid_count,
                    'Porcentaje_Invalido': np.round((invalid_count / total_rows * 100), 2)
                })
        
        df_resultados = pd.DataFrame(results)
        print(df_resultados)
        
        return df_resultados
    
    def impute_invalid_with_mode(self, df: pd.DataFrame) -> pd.DataFrame:

        df_imputed = df.copy()
        total_imputations = 0
        
        print("\n--- Iniciando Imputación de Consistencia Categórica (Usando la Moda) ---")
        
        for col, valid_values in self.config.validation_rules.items():
            if col in df_imputed.columns:
                invalid_mask = ~df_imputed[col].isin(valid_values)
                count_invalid = invalid_mask.sum()
                
                if count_invalid > 0:
                    imputation_mode_value = df_imputed[col].mode()
                    
                    if not imputation_mode_value.empty:
                        mode_value = imputation_mode_value.iloc[0]
                        df_imputed.loc[invalid_mask, col] = mode_value
                        total_imputations += count_invalid
                        print(f"   - Columna '{col}': {count_invalid} valores imputados con la moda ({mode_value}).")
                    else:
                        print(f"   - Columna '{col}': Advertencia: No se pudo calcular la moda.")
        
        print("-" * 40)
        return df_imputed


class OutlierHandler:
    
    def __init__(self, numeric_vars: list = None):
        self.numeric_vars = numeric_vars or ['laufzeit', 'wohnzeit', 'alter', 'bishkred', 'hoehe']
    
    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[
            (df[column] < lower_bound) | (df[column] > upper_bound)
        ][column]
        
        return outliers
    
    def impute_outliers_with_mean(self, df: pd.DataFrame) -> pd.DataFrame:

        df_imputed = df.copy()
        total_imputations = 0
        num_total = len(df_imputed)
        
        print("\n--- Iniciando Imputación de Outliers (Usando la Media) ---")
        
        for col in self.numeric_vars:
            if col in df_imputed.columns:
                outliers = self.detect_outliers_iqr(df_imputed, col)
                outlier_indices = outliers.index
                
                inlier_mask = ~df_imputed.index.isin(outlier_indices)
                num_imputed = len(outlier_indices)
                
                if num_imputed > 0:
                    unbiased_mean = df_imputed.loc[inlier_mask, col].mean()
                    
                    if pd.isna(unbiased_mean):
                        print(f"Advertencia: La media no pudo calcularse para la columna '{col}'.")
                    else:
                        df_imputed.loc[outlier_indices, col] = unbiased_mean
                        total_imputations += num_imputed
                        print(f"   - Columna '{col}': {num_imputed} valores imputados con la media ({unbiased_mean}). Modificaciones por renglón: {np.round(100 * (num_imputed/num_total), 2)}%")
                else:
                    print(f"   - Columna '{col}': No se encontraron valores atípicos para imputar.")
        
        print(f"Total general de valores imputados: {total_imputations}")
        print("-" * 40)
        
        return df_imputed
    
    def plot_outliers_boxplot(self, df: pd.DataFrame) -> None:

        df_numeric = df.select_dtypes(include=np.number)
        variables = [var for var in self.numeric_vars if var in df_numeric.columns]
        
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
        plt.show()
    
    def report_outliers(self, df: pd.DataFrame) -> dict:

        print("\n--- DETECCIÓN DE OUTLIERS (Método IQR) ---")
        n = len(df)
        outlier_counts = {}
        
        for var in self.numeric_vars:
            outliers_df = self.detect_outliers_iqr(df, var)
            if not outliers_df.empty:
                count = len(outliers_df)
                percentage = np.round(100 * count / n, 2)
                print(f"[{var}]: {count} outliers encontrados, {percentage}% del total")
                outlier_counts[var] = {'count': count, 'percentage': percentage}
        
        if not outlier_counts:
            print("No se detectaron outliers en las variables seleccionadas.")
        
        return outlier_counts


class DataVisualizer:
    
    def __init__(self, config: DataConfiguration):
        self.config = config
    
    def plot_descriptive_analysis(self, df: pd.DataFrame) -> None:

        # Boxplots para variables numéricas
        fig, axes = plt.subplots(1, 3, figsize=(35, 10))
        plt.subplots_adjust(wspace=0.1)
        axes = axes.ravel()
        for col, ax in zip(df[self.config.numerical_vars], axes):
            sns.boxplot(x=df[col], ax=ax, color='paleturquoise')
            ax.set(title=f'{col}', xlabel=None)
        plt.show()
        
        # Histogramas para variables ordinales
        fig, axes = plt.subplots(2, 3, figsize=(35, 15))
        fig.suptitle('Histograma de variables ordinales', fontsize=20, y=1)
        plt.subplots_adjust(wspace=0.4)
        axes = axes.ravel()
        
        for col, ax in zip(df[self.config.ordinal_vars], axes):
            sns.histplot(x=df[col], ax=ax, color='sandybrown', edgecolor='saddlebrown')
            ax.set(title=f'{col}', xlabel=None)
        plt.show()
        
        # Histogramas para variables categóricas
        fig, axes = plt.subplots(2, 6, figsize=(35, 15))
        fig.suptitle('Histograma de variables binarias y nominales', fontsize=20, y=1)
        plt.subplots_adjust(wspace=0.4, hspace=0.5)
        axes = axes.ravel()
        
        for col, ax in zip(df[self.config.categorical_vars], axes):
            sns.histplot(x=df[col], ax=ax, color='yellowgreen', edgecolor='darkolivegreen')
            ax.set(title=f'{col}', xlabel=None)
            ax.tick_params(axis='x', labelrotation=45)
        plt.xticks(rotation=45)
        plt.show()


class DataPreprocessor:
    
    def __init__(self, config: DataConfiguration):
        self.config = config
        self.preprocessor = self._build_preprocessor()
    
    def _build_preprocessor(self) -> ColumnTransformer:

        # Pipeline para variables numéricas
        numerical_pipe = Pipeline(steps=[
            ('impMediana', SimpleImputer(strategy='median')),
            ('escalaNum', MinMaxScaler(feature_range=(1, 2)))
        ])
        
        # Pipeline para variables categóricas
        nominal_pipe = Pipeline(steps=[
            ('impModa', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        # Pipeline para variables ordinales
        ordinal_pipe = Pipeline(steps=[
            ('impOrd', SimpleImputer(strategy='most_frequent')),
            ('ordtrasnf', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # Combinación de todos los transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('numpipe', numerical_pipe, self.config.numerical_vars),
                ('nominals', nominal_pipe, self.config.categorical_vars),
                ('ordinales', ordinal_pipe, self.config.ordinal_vars)
            ],
            remainder='passthrough'
        )
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:

        print("\n--- Iniciando Preprocesamiento de Datos ---")
        
        try:
            df_processed = self.preprocessor.fit_transform(df)
            
            print("Preprocesamiento completado. Datos listos para el modelo.")
            print(f"Shape de los datos transformados: {df_processed.shape}")
            
            return df_processed
        
        except Exception as e:
            print(f"Error durante el preprocesamiento: {e}")
            return None
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        return self.preprocessor.transform(df)
    
    def get_column_transformer(self) -> ColumnTransformer:
        """Get the column transformer instance."""
        return self.preprocessor


class DataSplitter:

    def split_by_target(
        self, 
        df: pd.DataFrame, 
        target_column_name: str = 'kredit',
        test_size: float = 0.3,
        random_state: int = 1234
    ) -> tuple:

        if target_column_name not in df.columns:
            print(f"Error: La columna objetivo '{target_column_name}' no se encontró en el DataFrame.")
            return None, None, None, None
        
        # Separar la variable objetivo y las features
        y = df[target_column_name].copy()
        X = df.drop(columns=[target_column_name], axis=1).copy()
        
        # Dividir los datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y, 
            random_state=random_state
        )
        
        print(f"División de datos completada.")
        print(f"   - X Train (Features): {X_train.shape[0]} filas, {X_train.shape[1]} columnas.")
        print(f"   - X Test (Features): {X_test.shape[0]} filas, {X_test.shape[1]} columnas.")
        print(f"   - y Train (Target '{target_column_name}'): {y_train.shape[0]} filas.")
        print(f"   - y Test (Target '{target_column_name}'): {y_test.shape[0]} filas.")
        
        return X_train, X_test, y_train, y_test


class ModelEvaluator:
    
    def __init__(self, preprocessor):

        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
    
    def evaluate_model(
        self, 
        modelo, 
        nombre: str, 
        X_train, 
        y_train, 
        metodo_balanceo=None,
        cv_splits: int = 5,
        cv_repeats: int = 3
    ) -> dict:

        pipeline = ImbPipeline(steps=[
            ('preprocesamiento', self.preprocessor),
            ('balanceo', metodo_balanceo),
            ('model', modelo)
        ])
        
        micv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=cv_repeats, random_state=5)
        mismetricas = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'gmean': make_scorer(geometric_mean_score)
        }
        
        scores = cross_validate(
            pipeline, X_train, np.ravel(y_train), 
            scoring=mismetricas, 
            cv=micv, 
            return_train_score=True
        )
        
        print(f'\n>> {nombre}')
        for j, k in enumerate(list(scores.keys())):
            if j > 1:  # Skip fit_time and score_time
                mean_val = np.nanmean(scores[k])
                std_val = np.nanstd(scores[k])
                print(f'\t {k}: {mean_val:.4f} ({std_val:.3f})')
        
        self.results[nombre] = scores
        return scores
    
    def get_default_models(self) -> dict:

        models = {
            'Logistic_Regression': LogisticRegression(
                penalty='l2',
                solver='newton-cg',
                max_iter=1000,
                C=1,
                random_state=1
            ),
            'Decision_Tree': DecisionTreeClassifier(
                max_depth=3,
                min_samples_split=20,
                random_state=123
            ),
            'Random_Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=3,
                min_samples_split=50,
                random_state=123
            ),
            'XGBoost': XGBClassifier(
                booster='gbtree',
                n_estimators=100,
                max_depth=3,
                learning_rate=0.01,
                subsample=0.7,
                random_state=5,
                n_jobs=-1
            )
        }
        
        self.models = models
        return models
    
    def evaluate_all_models(
        self, 
        X_train, 
        y_train, 
        use_balanceo: bool = True,
        balanceo_method: str = 'borderline'
    ) -> dict:

        models = self.get_default_models()
        
        if use_balanceo:
            if balanceo_method == 'borderline':
                metodo = BorderlineSMOTE(random_state=42, k_neighbors=5, m_neighbors=10)
            else:
                metodo = SMOTE(random_state=42)
        else:
            metodo = None
        
        for name, model in models.items():
            self.evaluate_model(model, name, X_train, y_train, metodo)
        
        return self.results


class SHAPExplainer:
    
    def __init__(self, preprocessor, model, X_train, y_train):

        self.preprocessor = preprocessor
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_preprocessed = None
        self.feature_names = None
        self.explainer = None
    
    def _prepare_data(self):
        # Preprocesar datos
        X_pre = self.preprocessor.transform(self.X_train)
        
        # Convertir a array denso si es sparse
        if hasattr(X_pre, 'toarray'):
            self.X_preprocessed = X_pre.toarray()
        else:
            self.X_preprocessed = np.asarray(X_pre)
        
        # Obtener nombres de features
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self):

        feature_names = []
        
        for name, transformer, cols in self.preprocessor.transformers_:
            if transformer == "drop":
                continue
            if transformer == "passthrough":
                if isinstance(cols, slice):
                    cols = self.X_train.columns[cols]
                feature_names.extend(list(cols))
                continue
            
            try:
                if hasattr(transformer, "get_feature_names_out"):
                    names = transformer.get_feature_names_out(cols)
                elif hasattr(transformer, "named_steps"):
                    last = list(transformer.named_steps.values())[-1]
                    names = last.get_feature_names_out(cols) if hasattr(last, "get_feature_names_out") else np.array(cols)
                else:
                    names = np.array(cols)
            except Exception:
                names = np.array(cols)
            
            feature_names.extend([str(n) for n in names])
        
        return feature_names
    
    def create_explainer(self, n_background_samples: int = 300):

        if self.X_preprocessed is None:
            self._prepare_data()
        
        shap.initjs()
        
        # Crear TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Submuestrear para velocidad si es necesario
        n_bg = min(n_background_samples, self.X_preprocessed.shape[0])
        idx = np.random.choice(self.X_preprocessed.shape[0], size=n_bg, replace=False)
        X_background = self.X_preprocessed[idx]
        
        # Calcular valores SHAP
        self.shap_values = self.explainer.shap_values(X_background)
        
        return self.shap_values, X_background
    
    def plot_summary(self, shap_values=None, X_background=None, show: bool = True):

        if shap_values is None or X_background is None:
            shap_values, X_background = self.create_explainer()
        
        plt.figure(figsize=(9, 5))
        shap.summary_plot(shap_values, X_background, feature_names=self.feature_names, show=False)
        plt.title("SHAP Summary (Global) — XGBoost")
        plt.tight_layout()
        
        if show:
            plt.show()
    
    def plot_dependence(self, feature_index: int = None, shap_values=None, X_background=None, show: bool = True):

        if shap_values is None or X_background is None:
            shap_values, X_background = self.create_explainer()
        
        if feature_index is None:
            # Encontrar la feature más importante
            importances = np.abs(shap_values).mean(axis=0)
            feature_index = np.argsort(importances)[-1]
        
        feature_name = self.feature_names[feature_index] if self.feature_names else f"f_{feature_index}"
        
        shap.dependence_plot(feature_name, shap_values, X_background, 
                           feature_names=self.feature_names, show=show)


class EDAWorkflow:
    
    def __init__(self, data_path: str = "../../data/raw/"):
        
        self.loader = DataLoader(data_path)
        self.config = DataConfiguration()
        self.cleaner = DataCleaner(self.config)
        self.outlier_handler = OutlierHandler(self.config.numerical_vars)
        self.visualizer = DataVisualizer(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.splitter = DataSplitter()
        self.model_evaluator = None
        self.shap_explainer = None
    
    def load_and_compare_datasets(self) -> tuple:

        df_original = self.loader.load_original_dataset()
        df_modified = self.loader.load_modified_dataset()
        
        self.loader.compare_datasets(df_original, df_modified)
        
        return df_original, df_modified
    
    def execute_full_pipeline(self, df: pd.DataFrame, target_column: str = 'kredit') -> dict:

        results = {}
        
        # Paso 1: Limpiar valores numéricos inválidos
        print("\n=== PASO 1: Limpieza de Valores Numéricos Inválidos ===")
        df_clean = self.cleaner.clean_invalid_numeric_values(df)
        results['cleaned_data'] = df_clean
        
        # Paso 2: Limpiar nulos en la variable objetivo
        print("\n=== PASO 2: Limpieza de Nulos en la Variable Objetivo ===")
        df_clean_target = self.cleaner.clean_target_nulls(df_clean, target_column)
        results['clean_target'] = df_clean_target
        
        # Paso 3: Imputar valores categóricos inválidos
        print("\n=== PASO 3: Imputación de Valores Categóricos Inválidos ===")
        df_imputed = self.cleaner.impute_invalid_with_mode(df_clean_target)
        results['imputed_categorical'] = df_imputed
        
        # Paso 4: Manejar outliers
        print("\n=== PASO 4: Manejo de Outliers ===")
        df_imputed_outliers = self.outlier_handler.impute_outliers_with_mean(df_imputed)
        results['imputed_outliers'] = df_imputed_outliers
        
        # Paso 5: Eliminar columna mixed_type_col (pendiente de análisis)
        print("\n=== PASO 5: Eliminando columna mixed_type_col ===")
        if 'mixed_type_col' in df_imputed_outliers.columns:
            df_final = df_imputed_outliers.drop('mixed_type_col', axis=1)
            print(f"✓ Columna 'mixed_type_col' eliminada. Shape actual: {df_final.shape}")
        else:
            df_final = df_imputed_outliers
            print("La columna 'mixed_type_col' no existe en el dataset")
        results['data_final'] = df_final
        
        # Paso 6: Generar visualizaciones
        print("\n=== PASO 6: Generación de Visualizaciones ===")
        self.visualizer.plot_descriptive_analysis(df_final)
        results['visualizations'] = 'Completo'
        
        # Paso 7: Dividir datos
        print("\n=== PASO 7: División de Datos ===")
        X_train, X_test, y_train, y_test = self.splitter.split_by_target(
            df_final, target_column=target_column
        )
        
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        
        # Paso 8: Preprocesar datos
        print("\n=== PASO 8: Preprocesamiento de Datos ===")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        results['X_train_processed'] = X_train_processed
        results['X_test_processed'] = X_test_processed
        results['preprocessor'] = self.preprocessor
        
        return results
    
    def evaluate_models(
        self, 
        X_train, 
        y_train, 
        use_balanceo: bool = True,
        balanceo_method: str = 'borderline'
    ) -> dict:

        # Crear evaluador si no existe
        if self.model_evaluator is None:
            self.model_evaluator = ModelEvaluator(self.preprocessor.get_column_transformer())
        
        print("\n=== EVALUACIÓN DE MODELOS ===")
        
        results = self.model_evaluator.evaluate_all_models(
            X_train, y_train, use_balanceo, balanceo_method
        )
        
        return results
    
    def create_shap_explanations(
        self, 
        best_model, 
        X_train, 
        y_train,
        plot_summary: bool = True,
        plot_dependence: bool = True
    ):

        print("\n=== EXPLICABILIDAD CON SHAP ===")
        
        # Crear explainer
        self.shap_explainer = SHAPExplainer(
            self.preprocessor.get_column_transformer(),
            best_model,
            X_train,
            y_train
        )
        
        # Calcular valores SHAP
        shap_values, X_background = self.shap_explainer.create_explainer()
        
        # Generar plots
        if plot_summary:
            print("\n--- Generando SHAP Summary Plot ---")
            self.shap_explainer.plot_summary(shap_values, X_background)
        
        if plot_dependence:
            print("\n--- Generando SHAP Dependence Plot ---")
            self.shap_explainer.plot_dependence(None, shap_values, X_background)
        
        return shap_values, X_background

