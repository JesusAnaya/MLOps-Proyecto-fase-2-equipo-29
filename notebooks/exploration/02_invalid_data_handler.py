import pandas as pd
import numpy as np

class InvalidDataHandler: 

    """
    Clase para detectar valores no válidos dentro de nuestra BD y realizar el proceso de imputación 
    de los datos inconsistentes a través de la moda (para el caso categórico)
    """

    def __init__(self, target_column_name: str,  valid_values_map: dict, mixed_type_col: str = None):

        self.rules = valid_values_map
        self.valid_rules = valid_values_map
        self.mixed_col = mixed_type_col
        ## Detect the columns we're about to clean
        self.cols_to_impute = list(valid_values_map.keys())


    def clean_and_transform(self, df_input : pd.DataFrame) -> pd.DataFrame: 

        """
        Método para realizar la limpieza de los datos que no son válidos dentro de nuestro dataframe

        """
        df_transformed = df_input.copy()

        ## Conversión de los datos numéricos
        df_transformed = self._apply_numeric_conversion_logic(df_transformed)

        ## Limpieza particular para la variable objetivo
        df_transformed = self._apply_drop_logic(df_transformed)
        
        ## Limpieza para las variables features (variables características)
        df_transformed = self._apply_impute_logic(df_transformed)

        return df_transformed
    
    def _apply_numeric_conversion_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Identifica y reemplaza elementos no numéricos atípicos por NaN, 
        luego convierte las columnas relevantes a tipo numérico.
        """

        data_temp = df.copy()

        non_numeric_summary = {}

        # 1. Detección de elementos no numéricos
        for col in data_temp.columns:
            # Intentar convertir a numérico, forzando errores a NaN
            temp_numeric = pd.to_numeric(data_temp[col], errors='coerce')
            non_numeric_elements = temp_numeric.isna()

            if non_numeric_elements.any():
                non_numeric_values = data_temp[col][non_numeric_elements]
                # Solo contamos los elementos que *realmente* eran texto o caracteres atípicos
                non_numeric_summary[col] = non_numeric_values.value_counts().to_dict()

        non_numeric_series = pd.Series(non_numeric_summary)

        # 2. Identificar elementos a substituir por NaN (excluyendo la columna de tipo mixto)
        all_unique_non_numeric_elements = []
        for col, count_dict in non_numeric_series.items():
            if col != self.mixed_col:
                 all_unique_non_numeric_elements.extend(count_dict.keys())

        # 3. Reemplazar los valores por NaN
        final_elements = list(set(all_unique_non_numeric_elements))
        data_clean = data_temp.replace(final_elements, np.nan)
        
        # 4. Conversión final a tipo numérico
        columnas_numericas = data_clean.columns.tolist()
        if self.mixed_col and self.mixed_col in columnas_numericas:
            columnas_numericas.remove(self.mixed_col)

        data_clean[columnas_numericas] = data_clean[columnas_numericas].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        )
        
        return data_clean


    def _apply_drop_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Lógica para eliminar las filas que no tienen valores válidos. 
        Debido a que la variable objetivo que tenemos es uno de los elementos más importantes en el proceso de desarrollo del modelo, 
        en lugar de imputarla, como al resto de las variables, eliminaremos los valores no válidos, con el objetivo de no incluir ruido en nuestro 
        modelo.
        """

        target_col = self.target_col

        if target_col not in df.columns: 
            print(f"Error (Drop logic): La columna objetivo {target_col} no existe en el DataFrame.")
            return df.copy()

        df_clean_target = df[(~df[target_col].isnull()) & (df[target_col].isin([0, 1]))].copy()

        return df_clean_target

    def _apply_impute_logic(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Lógica para imputar los valores no válidos. Con la moda para el caso de las variables categóricas. 
        Nota: Se realiza la imputación de datos, pues el EDA arrojo que eliminar las variables nos dejaría con muy poca información para el entrenamiento. 
        """
        df_imputed = df.copy()

        for col, valid_values in self.valid_rules.items():
            if col in df_imputed.columns: 
                invalid_mask = ~ df_imputed[col].isin(valid_values)
                count_invalid = invalid_mask.sum()

                if count_invalid > 0: 
                    imputation_mode_value = df_imputed[col].mode()

                    if not imputation_mode_value.empty:
                        mode_value = imputation_mode_value.iloc[0]

                        ## Reemplazar los valores inválidos por la moda
                        df_imputed.loc[invalid_mask, col] = mode_value

        return df_imputed