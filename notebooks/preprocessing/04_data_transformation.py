import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

class DataPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """
    Clase que encapsula el ColumnTransformer de scikit-learn para el preprocesamiento
    de datos: imputación, escalado y codificación (OHE y Ordinal).
    """

    def __init__(self, numericas_cols: list, nominales_cols: list, ordinales_cols: list):
        """
        Inicializa el transformador con los nombres de las columnas.

        Args:
            numericas_cols (list): Columnas numéricas.
            nominales_cols (list): Columnas categóricas nominales (para OHE).
            ordinales_cols (list): Columnas categóricas ordinales (para Ordinal Encoding).
        """
        self.numericas_cols = numericas_cols
        self.nominales_cols = nominales_cols
        self.ordinales_cols = ordinales_cols
        self.preprocessor = None 
        self._build_preprocessor()

    def _build_preprocessor(self):
        """
        Método privado para construir las pipelines de scikit-learn y el ColumnTransformer.
        """
        
        # 1. Pipeline para Variables Numéricas: Mediana + MinMax (1, 2)
        numericas_pipe = Pipeline(steps=[
            ('impMediana', SimpleImputer(strategy='median')),
            ('escalaNum', MinMaxScaler(feature_range=(1, 2)))
        ])

        # 2. Pipeline para Variables Nominales: Moda + One-Hot Encoding
        nominales_pipe = Pipeline(steps=[
            ('impModa', SimpleImputer(strategy='most_frequent')),
            # drop='first' para evitar la trampa de la variable ficticia
            ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore')) 
        ])

        # 3. Pipeline para Variables Ordinales: Moda + Ordinal Encoding
        ordinales_pipe = Pipeline(steps=[
            ('impOrd', SimpleImputer(strategy='most_frequent')),
            # unknown_value=-1 permite manejar categorías que no estaban en el fit
            ('ordtrasnf', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # 4. Combinación de Pipelines con ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numericas_pipe, self.numericas_cols),
                ('nom', nominales_pipe, self.nominales_cols),
                ('ord', ordinales_pipe, self.ordinales_cols)
            ],
            remainder='passthrough', # Mantiene otras columnas sin procesar
            verbose_feature_names_out=False # Permite obtener nombres de columnas limpios
        ).set_output(transform="pandas") # IMPORTANTE: Devuelve un DataFrame de Pandas

        print("ColumnTransformer interno construido.")

    def fit(self, X: pd.DataFrame, y=None):
        """
        Ajusta el ColumnTransformer interno a los datos de entrenamiento.
        """
        try:
            self.preprocessor.fit(X)
            print("Ajuste completado. El preprocesador está listo para transformar.")
            return self
        except Exception as e:
            print(f"Error durante el ajuste (FIT): {e}")
            raise


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones aprendidas a un nuevo conjunto de datos.
        """
 
        try:
            if self.preprocessor is None:
                raise RuntimeError("El preprocesador no ha sido construido. Llame a __init__.")
            
            # Solo aplica la transformación, usando los estadísticos aprendidos en fit
            X_transformed = self.preprocessor.transform(X)
            
            print("Transformación completada. Datos listos para el modelo.")
            print(f"Shape de los datos transformados: {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            print(f"Error durante la transformación: {e}")
            raise

