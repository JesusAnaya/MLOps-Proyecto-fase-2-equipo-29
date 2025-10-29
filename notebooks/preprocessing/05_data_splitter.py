import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

class DataSplitter:
    """
    Clase para dividir un DataFrame en conjuntos de entrenamiento y prueba (X/y).

    Los hiperparámetros de la división (tamaño de prueba y semilla aleatoria) 
    se configuran en el momento de la instanciación de la clase.
    """
    def __init__(self, test_size: float = 0.3, random_state: int = 1234):

        # Atributos internos de la clase (configuraciones)
        self.test_size = test_size
        self.random_state = random_state
        print(f"DataSplitter inicializado con: test_size={self.test_size}, random_state={self.random_state}")

    def split_data_by_target(self, 
                             df: pd.DataFrame, 
                             target_column_name: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.Series], Optional[pd.Series]]:

        # Verificar si la columna target existe en el DataFrame
        if target_column_name not in df.columns:
            print(f"Error: La columna objetivo '{target_column_name}' no se encontró en el DataFrame.")
            return None, None, None, None

        # 1. Separar X (features) e y (target)
        y = df[target_column_name].copy()
        X = df.drop(columns=[target_column_name], axis=1).copy()

        # 2. Aplicar la división de entrenamiento/prueba usando los atributos de la clase
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, 
                y, 
                test_size=self.test_size, 
                stratify=y, # Mantiene la proporción de la clase objetivo en ambos sets
                random_state=self.random_state
            )
        except ValueError as e:
            # Capturar errores comunes como cuando 'y' tiene muy pocos valores únicos para estratificar
            print(f"Error al aplicar train_test_split (ValueError): {e}")
            return None, None, None, None

        print(f"\nDivisión de datos completada usando test_size={self.test_size}.")
        print(f"   - X Train (Features): {X_train.shape[0]} filas, {X_train.shape[1]} columnas.")
        print(f"   - X Test (Features): {X_test.shape[0]} filas, {X_test.shape[1]} columnas.")
        print(f"   - y Train (Target '{target_column_name}'): {y_train.shape[0]} filas.")
        print(f"   - y Test (Target '{target_column_name}'): {y_test.shape[0]} filas.")

        return X_train, X_test, y_train, y_test