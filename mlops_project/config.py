"""
Configuration module for MLOps project.

Este módulo centraliza todas las configuraciones del proyecto:
- Rutas de datos
- Hiperparámetros del modelo
- Parámetros de preprocesamiento
- Semillas aleatorias para reproducibilidad
"""

from pathlib import Path
from typing import Dict, List, Tuple

# === Rutas del proyecto ===
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# === Configuración de datos ===
TARGET_COLUMN = "kredit"
MIXED_TYPE_COLUMN = "mixed_type_col"  # Columna a eliminar del dataset original

# Archivos de datos
RAW_DATA_ORIGINAL = "german_credit_original.csv"
RAW_DATA_MODIFIED = "german_credit_modified.csv"

PROCESSED_X_TRAIN_TEST = "Xtraintest.csv"
PROCESSED_Y_TRAIN_TEST = "ytraintest.csv"
PROCESSED_DATA_CLEAN = "data_clean.csv"

# === Semilla aleatoria para reproducibilidad ===
RANDOM_SEED = 42

# === Configuración de división de datos ===
TEST_SIZE = 0.3
STRATIFY = True  # Mantener proporción de clases en train/test

# === Definición de tipos de variables ===
NUMERIC_FEATURES: List[str] = [
    "laufzeit",  # duration - quantitative
    "hoehe",  # amount - quantitative
    "alter",  # age - quantitative
]

ORDINAL_FEATURES: List[str] = [
    "beszeit",  # employment_duration - ordinal
    "rate",  # installment_rate - ordinal
    "wohnzeit",  # present_residence - ordinal
    "verm",  # property - ordinal
    "bishkred",  # number_credits - ordinal
    "beruf",  # job - ordinal
]

NOMINAL_FEATURES: List[str] = [
    "laufkont",  # status - categorical
    "moral",  # credit_history - categorical
    "verw",  # purpose - categorical
    "sparkont",  # savings - categorical
    "famges",  # personal_status_sex - categorical
    "buerge",  # other_debtors - categorical
    "weitkred",  # other_installment_plans - categorical
    "wohn",  # housing - categorical
    "pers",  # people_liable - binary
    "telef",  # telephone - binary
    "gastarb",  # foreign_worker - binary
]

# === Reglas de validación para variables categóricas ===
CATEGORICAL_VALIDATION_RULES: Dict[str, List[int]] = {
    # Variables nominales
    "laufkont": [1, 2, 3, 4],
    "moral": [0, 1, 2, 3, 4],
    "verw": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10],
    "sparkont": [1, 2, 3, 4, 5],
    "famges": [1, 2, 3, 4],
    "buerge": [1, 2, 3],
    "weitkred": [1, 2, 3],
    "wohn": [1, 2, 3],
    # Variables ordinales
    "beszeit": [1, 2, 3, 4, 5],
    "rate": [1, 2, 3, 4],
    "wohnzeit": [1, 2, 3, 4],
    "verm": [1, 2, 3, 4],
    "bishkred": [1, 2, 3, 4],
    "beruf": [1, 2, 3, 4],
    # Variables binarias
    "pers": [1, 2],
    "telef": [1, 2],
    "gastarb": [1, 2],
    # Variable objetivo
    "kredit": [0, 1],
}

# === Configuración de detección de outliers ===
OUTLIER_METHOD = "IQR"  # Opciones: 'IQR' o 'Percentiles'
OUTLIER_PERCENTILES: Tuple[float, float] = (0.05, 0.95)  # Solo si method='Percentiles'
OUTLIER_VARIABLES: List[str] = ["laufzeit", "wohnzeit", "alter", "bishkred", "hoehe"]

# === Configuración de preprocesamiento ===
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"
NUMERIC_SCALER_RANGE: Tuple[int, int] = (1, 2)  # MinMaxScaler range

# === Configuración de validación cruzada ===
CV_FOLDS = 5
CV_REPEATS = 3
CV_METHOD = "RepeatedStratifiedKFold"

# === Configuración de modelos ===
MODEL_CONFIG: Dict[str, any] = {
    "cv_folds": CV_FOLDS,
    "cv_repeats": CV_REPEATS,
    "random_state": RANDOM_SEED,
}

# Hiperparámetros del mejor modelo (Logistic Regression)
BEST_MODEL_PARAMS: Dict[str, any] = {
    "penalty": "l2",
    "solver": "newton-cg",
    "max_iter": 1000,
    "C": 1,
    "random_state": RANDOM_SEED,
}

# Configuración de SMOTE para balanceo de clases
SMOTE_CONFIG: Dict[str, any] = {
    "method": "BorderlineSMOTE",  # Opciones: 'SMOTE', 'BorderlineSMOTE', 'ADASYN'
    "random_state": RANDOM_SEED,
    "k_neighbors": 5,
    "m_neighbors": 10,
}

# === Métricas de evaluación ===
EVALUATION_METRICS: List[str] = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]

# === Configuración de modelos disponibles ===
AVAILABLE_MODELS: Dict[str, Dict] = {
    "logistic_regression": {"name": "Logistic Regression", "params": BEST_MODEL_PARAMS},
    "decision_tree": {
        "name": "Decision Tree",
        "params": {"max_depth": 3, "min_samples_split": 20, "random_state": RANDOM_SEED},
    },
    "random_forest": {
        "name": "Random Forest",
        "params": {
            "n_estimators": 200,
            "max_depth": 3,
            "min_samples_split": 50,
            "random_state": RANDOM_SEED,
        },
    },
    "svm": {
        "name": "Support Vector Machine",
        "params": {"kernel": "rbf", "C": 10, "gamma": "auto", "random_state": RANDOM_SEED},
    },
    "xgboost": {
        "name": "XGBoost",
        "params": {
            "booster": "gbtree",
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.01,
            "subsample": 0.7,
            "random_state": RANDOM_SEED,
            "n_jobs": -1,
        },
    },
}

# === Nombres de archivos de salida ===
BEST_MODEL_FILENAME = "best_model.joblib"
PREPROCESSOR_FILENAME = "preprocessor.joblib"
RESULTS_FILENAME = "model_results.json"


def ensure_directories() -> None:
    """
    Crea los directorios necesarios si no existen.
    """
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INTERIM_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_data_path(filename: str, data_type: str = "raw") -> Path:
    """
    Retorna la ruta completa para un archivo de datos.

    Args:
        filename: Nombre del archivo
        data_type: Tipo de datos ('raw', 'processed', 'interim', 'external')

    Returns:
        Path completo del archivo
    """
    data_type_mapping = {
        "raw": RAW_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
        "interim": INTERIM_DATA_DIR,
        "external": EXTERNAL_DATA_DIR,
    }

    if data_type not in data_type_mapping:
        raise ValueError(f"data_type debe ser uno de: {list(data_type_mapping.keys())}")

    return data_type_mapping[data_type] / filename


def get_model_path(filename: str) -> Path:
    """
    Retorna la ruta completa para un archivo de modelo.

    Args:
        filename: Nombre del archivo del modelo

    Returns:
        Path completo del archivo
    """
    return MODELS_DIR / filename


# === Inicialización ===
# Crear directorios al importar el módulo
ensure_directories()


if __name__ == "__main__":
    # Imprimir configuración para verificación
    print("=== Configuración del Proyecto MLOps ===")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Target Column: {TARGET_COLUMN}")
    print("\nFeatures:")
    print(f"  - Numeric: {len(NUMERIC_FEATURES)}")
    print(f"  - Ordinal: {len(ORDINAL_FEATURES)}")
    print(f"  - Nominal: {len(NOMINAL_FEATURES)}")
    print(
        f"\nTotal Features: {len(NUMERIC_FEATURES) + len(ORDINAL_FEATURES) + len(NOMINAL_FEATURES)}"
    )
    print(f"\nCross-Validation: {CV_FOLDS} folds, {CV_REPEATS} repeats")
    print("Best Model: Logistic Regression")
    print(f"SMOTE Method: {SMOTE_CONFIG['method']}")
