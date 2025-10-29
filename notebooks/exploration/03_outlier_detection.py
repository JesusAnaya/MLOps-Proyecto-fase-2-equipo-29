import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class OutlierHandler: 
    
    def __init__(self, method: str, cap_percentiles: tuple, variables_to_cap : list):
        """
        Inicializa el transformador con la configuración de detección y delimitación.
            
        Args:
            method (str): Método de detección/delimitación: 'IQR' o 'Percentiles'.
            cap_percentiles (tuple): Par de percentiles (P_low, P_high) para el método 'Percentiles' (e.g., (0.05, 0.95)).
            variables_to_cap (list): Lista de columnas numéricas a las que se aplicará la delimitación.
        """
        if method not in ['IQR', 'Percentiles']:
            raise ValueError("El método debe ser 'IQR' o 'Percentiles'")
            
        self.method = method
        self.cap_percentiles = cap_percentiles
        self.variables_to_cap = variables_to_cap
        self.outlier_limit_ = {} # Atributo para almacenar los límites calculados (Q1, Q3, etc.)

        def fit(self, df_train: pd.DataFrame):
            """
            Calcula y almacena los límites de los outliers (lower_bound y upper_bound) 
            basados en el DataFrame de entrenamiento.
            """
            df_numeric = df_train[self.variables_to_cap].select_dtypes(include = np.number)

            for var in self.variables_to_cap:
                if var not in df_numeric.columns: 
                    print(f"Advertencia: '{var}' no es un valor numérico, o no se encuentra dentro de la base. Ignorado")
                    continue
                
                if self.method == 'IQR': 
                    Q1 = df_numeric[var].quantile(0.25)
                    Q3 = df_numeric[var].quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR

                elif self.method == 'Percentiles': 
                    p_low, p_high = self.cap_percentiles
                    lower_bound = df_numeric[var].quantile(p_low)
                    upper_bound = df_numeric[var].quantile(p_high)

                self.outlier_limit_[var] = {
                    'lower': lower_bound,
                    'upper':upper_bound
                }

            def transform(self, df_target:pd.DataFrame) -> pd.DataFrame: 
                """
                Aplica la delimitación (capping) a los outliers en el DataFrame objetivo 
                utilizando los límites calculados en 'fit'.
                """

                df_capped = df_target.copy()
                total_caps = 0
                
                for var, limits in self.outlier_limit_.items():
                    lower = limits['lower']
                    upper = limits['upper']
                    
                    # Identificar outliers
                    is_lower_outlier = (df_capped[var] < lower)
                    is_upper_outlier = (df_capped[var] > upper)
                    
                    count_caps = is_lower_outlier.sum() + is_upper_outlier.sum()
                    total_caps += count_caps
                    
                    if count_caps > 0:
                        # Aplicar capping: reemplazar outliers por el límite
                        df_capped.loc[is_lower_outlier, var] = lower
                        df_capped.loc[is_upper_outlier, var] = upper
                        # print(f"  - {var}: {count_caps} valores delimitados.")
                
                return df_capped

            def generate_boxplot(self, df: pd.DataFrame, title: str = "Boxplots de Variables Numéricas") -> None:
                    """
                    Crea un gráfico de caja para visualizar la distribución y los outliers.
                    (Adaptado de la función 'plot_output').

                    Args:
                        df (pd.DataFrame): DataFrame a graficar.
                        title (str): Título principal del gráfico.
                    """
                    variables = [var for var in self.variables_to_cap if var in df.columns]
                    df_numeric = df[variables].select_dtypes(include=np.number)
                    num_vars = len(variables)

                    if num_vars == 0:
                        print("No hay variables numéricas seleccionadas para graficar.")
                        return

                    n_cols = 5
                    n_rows = (num_vars + n_cols - 1) // n_cols

                    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

                    for i, var in enumerate(variables):
                        ax = plt.subplot(n_rows, n_cols, i + 1)
                        sns.boxplot(y=df_numeric[var], ax=ax, color='#6B8E23') # Color Olive Drab
                        ax.set_title(var, fontsize=12)
                        ax.set_ylabel('')
                        ax.set_xlabel('')
                        ax.tick_params(axis='y', labelsize=10)
                        ax.grid(axis='y', linestyle='--', alpha=0.5)

                    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
                    plt.suptitle(f"{title} ({num_vars} Variables)", fontsize=20, y=1.0)
                    plt.show()