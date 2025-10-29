# Proyecto MLOps - Equipo 29

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto de MLOps para la materia TC5044.10 - Implementación de pipeline automatizado para clasificación de crédito bancario.

## Información del Equipo

**Equipo:** 29  
**Materia:** Machine Learning Operations

### Integrantes del Equipo:
- Jesús Armando Anaya Orozco
- Oliver Josué De León Milian
- Isaura Yutsil Flores Escamilla
- Ovidio Alejandro Hernández Ruano
- Owen Jáuregui Borbón

## Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Uso](#uso)
- [Scripts Disponibles](#scripts-disponibles)
- [Pipeline MLOps](#pipeline-mlops)
- [Testing](#testing)
- [Configuración](#configuración)
- [Próximos Pasos](#próximos-pasos)

## Descripción del Proyecto

Este proyecto implementa un pipeline MLOps completo para el análisis y predicción de riesgo crediticio usando el dataset South German Credit. El enfoque está en crear una infraestructura robusta y automatizada para el ciclo de vida de modelos de machine learning.

### Características Principales

- Pipeline automatizado de preparación de datos
- Ingeniería de features con validación y transformación
- Entrenamiento de modelos con cross-validation
- Sistema de predicción con evaluación de métricas
- Tests comprehensivos (130 tests, 64% coverage)
- Scripts CLI para automatización
- Configuración centralizada
- Preparado para integración con MLFlow

## Requisitos

- **Python**: 3.12.0
- **UV**: Gestor de paquetes Python (instalado localmente)
  - UV maneja el entorno virtual automáticamente
  - No es necesario activar el venv manualmente
- **Sistema Operativo**: macOS, Linux, o Windows

## Instalación

**Importante**: Este proyecto usa **UV** para gestión de paquetes. No es necesario usar `pip` o activar entornos virtuales manualmente. UV gestiona todo automáticamente.

### 1. Clonar el Repositorio

```bash
git clone git@github.com:JesusAnaya/MLOps-Proyecto-fase-2-equipo-29.git
cd MLOps-Proyecto-fase-2-equipo-29
```

### 2. Inicializar el Proyecto

```bash
# Método recomendado: Inicialización completa con UV
make init

# O paso a paso:
make create-environment  # Crea venv con UV
make requirements        # Instala dependencias con uv sync
```

### 3. Uso con UV (Recomendado)

**UV permite ejecutar comandos directamente sin activar el entorno virtual**:

```bash
# Ejecutar scripts directamente
uv run mlops-prepare-data --help

# Ejecutar tests
uv run pytest tests/

# Ejecutar Python con el entorno UV
uv run python script.py
```

**Opcional: Activar el Entorno Virtual** (si prefieres trabajar en el shell):

```bash
# Unix/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Nota: Con UV, no es necesario activar el entorno virtual. Puedes usar `uv run` para ejecutar cualquier comando Python.

### 4. Verificar Instalación

```bash
# Ver scripts disponibles
make scripts

# Ejecutar tests
make test
```

## Ventajas de UV

Este proyecto usa **UV** como gestor de paquetes en lugar de pip/virtualenv tradicional:

### Ventajas de UV

1. **Sin Activación de Entorno Virtual**: UV maneja el entorno automáticamente
   ```bash
   # No necesitas: source .venv/bin/activate
   # Simplemente usa:
   uv run python script.py
   uv run pytest tests/
   uv run mlops-prepare-data --help
   ```

2. **Comandos Más Rápidos**: UV es significativamente más rápido que pip
   ```bash
   uv sync           # Instalar dependencias (más rápido que pip install)
   uv add pandas     # Agregar dependencia (más rápido que pip install)
   ```

3. **Gestión Automática del Entorno**: UV crea y gestiona el venv automáticamente
   ```bash
   uv venv --python 3.12  # Crea venv (si no existe)
   uv sync                # Crea venv automáticamente si es necesario
   ```

4. **Compatibilidad Total**: UV es compatible con pyproject.toml y pip
   ```bash
   # Todos los comandos tradicionales funcionan con uv run
   uv run python -m pytest
   uv run python -m black .
   ```

### Equivalencias Comunes

| Tradicional | Con UV |
|-------------|--------|
| `python script.py` | `uv run python script.py` |
| `pytest tests/` | `uv run pytest tests/` |
| `pip install package` | `uv add package` |
| `pip install -e .` | `uv sync` |
| `pip list` | `uv pip list` |
| `source .venv/bin/activate` | No necesario (usar `uv run`) |

## Estructura del Proyecto

```
proyecto_etapa_2/
│
├── mlops_project/              # Código fuente principal
│   ├── __init__.py
│   ├── config.py              # Configuración centralizada
│   ├── dataset.py             # Carga y preparación de datos
│   ├── features.py            # Ingeniería de features
│   ├── plots.py               # Visualizaciones
│   ├── modeling/              # Módulo de modelado
│   │   ├── __init__.py
│   │   ├── train.py          # Entrenamiento de modelos
│   │   └── predict.py        # Predicción e inferencia
│   └── README.md             # Guía de orquestación
│
├── tests/                     # Tests automatizados
│   ├── conftest.py           # Fixtures compartidas
│   ├── test_config.py        # Tests de configuración
│   ├── test_dataset.py       # Tests de datos
│   ├── test_features.py      # Tests de features
│   ├── test_modeling.py      # Tests de modelado
│   └── ...                   # Tests adicionales
│
├── data/                      # Datos del proyecto
│   ├── raw/                  # Datos originales
│   ├── processed/            # Datos procesados
│   ├── interim/              # Datos intermedios
│   └── external/             # Datos externos
│
├── models/                    # Modelos entrenados
├── notebooks/                 # Notebooks de exploración
├── reports/                   # Reportes y figuras
│
├── pyproject.toml            # Configuración del proyecto
├── Makefile                  # Comandos automatizados
└── README.md                 # Este archivo
```

## Uso

### Comandos Rápidos

```bash
# Ver todos los comandos disponibles
make help

# Ejecutar pipeline completo
make pipeline

# Verificar calidad del código
make check

# Ver scripts disponibles
make scripts
```

## Scripts Disponibles

El proyecto incluye 4 scripts CLI principales para automatizar el pipeline MLOps:

### 1. `mlops-prepare-data` - Preparación de Datos

**Función**: Carga datos crudos, limpia valores inválidos, elimina outliers y divide en train/test.

```bash
# Uso básico
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save

# O con make
make prepare-data
```

**Opciones**:
- `--input`: Ruta al archivo CSV de entrada (requerido)
- `--save`: Guardar datos procesados en data/processed/
- `--combined`: Retornar datos combinados sin dividir

**Salida**:
- `data/processed/data_clean.csv`: Datos limpios
- `data/processed/Xtraintest.csv`: Features combinadas
- `data/processed/ytraintest.csv`: Variable objetivo

### 2. `mlops-prepare-features` - Ingeniería de Features

**Función**: Transforma features aplicando imputación, escalado y encoding.

```bash
# Uso básico
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor

# Con datos de test
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --test data/processed/X_test.csv \
    --output-train X_train_transformed.csv \
    --output-test X_test_transformed.csv \
    --save-preprocessor

# O con make
make prepare-features
```

**Opciones**:
- `--train`: Archivo CSV con datos de entrenamiento (requerido)
- `--test`: Archivo CSV con datos de prueba (opcional)
- `--output-train`: Archivo de salida para train (default: X_train_processed.csv)
- `--output-test`: Archivo de salida para test (default: X_test_processed.csv)
- `--save-preprocessor`: Guardar el preprocessor en models/

**Salida**:
- Features transformadas
- `models/preprocessor.joblib`: Preprocessor guardado

### 3. `mlops-train` - Entrenamiento de Modelos

**Función**: Entrena modelos con cross-validation y guarda el mejor modelo.

```bash
# Uso básico (Logistic Regression)
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression

# Entrenar Random Forest sin SMOTE
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model random_forest \
    --no-smote

# Sin evaluación (solo entrenar)
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression \
    --no-evaluate

# O con make
make train
```

**Opciones**:
- `--X-train`: Features de entrenamiento (requerido)
- `--y-train`: Variable objetivo de entrenamiento (requerido)
- `--preprocessor`: Ruta al preprocessor guardado (requerido)
- `--model`: Modelo a entrenar (default: logistic_regression)
  - Opciones: `logistic_regression`, `random_forest`, `decision_tree`, `svm`, `xgboost`
- `--no-smote`: No usar SMOTE para balanceo de clases
- `--smote-method`: Método de SMOTE (default: BorderlineSMOTE)
  - Opciones: `SMOTE`, `BorderlineSMOTE`
- `--no-evaluate`: No evaluar con cross-validation
- `--output`: Nombre del archivo de salida para el modelo

**Salida**:
- `models/best_model.joblib`: Modelo entrenado
- `models/model_results.json`: Métricas de evaluación

### 4. `mlops-predict` - Predicción e Inferencia

**Función**: Realiza predicciones con un modelo entrenado y evalúa métricas.

```bash
# Predicción con evaluación
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save

# Solo predicción (sin evaluación)
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_new.csv \
    --output predictions.csv \
    --save

# Predicción con probabilidades
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_test.csv \
    --proba \
    --save

# Predicción en batches grandes
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/X_large.csv \
    --batch-size 1000 \
    --save

# O con make
make predict
```

**Opciones**:
- `--model`: Ruta al modelo guardado (default: models/best_model.joblib)
- `--X-test`: Features de test (requerido)
- `--y-test`: Variable objetivo de test (opcional, para evaluación)
- `--output`: Archivo de salida para predicciones (default: predictions.csv)
- `--batch-size`: Tamaño del batch para predicciones (default: 1000)
- `--proba`: Guardar probabilidades en lugar de clases
- `--save`: Guardar predicciones en archivo

**Salida**:
- `predictions.csv`: Predicciones guardadas
- `evaluation_metrics.json`: Métricas de evaluación (si se provee y_test)

## Pipeline MLOps

### Pipeline Completo Automatizado

```bash
# Ejecutar todo el pipeline de una vez
make pipeline
```

Este comando ejecuta secuencialmente:
1. `make prepare-data` - Preparación de datos
2. `make prepare-features` - Transformación de features
3. `make train` - Entrenamiento del modelo

### Pipeline Manual Paso a Paso

```bash
# 1. Preparar datos
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save

# 2. Preparar features
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor

# 3. Entrenar modelo
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression

# 4. Realizar predicciones
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
```

### Ejemplo: Entrenar Múltiples Modelos

```bash
# Iterar sobre diferentes modelos
for model in logistic_regression random_forest decision_tree xgboost; do
    echo "Entrenando $model..."
    uv run mlops-train \
        --X-train data/processed/Xtraintest.csv \
        --y-train data/processed/ytraintest.csv \
        --preprocessor models/preprocessor.joblib \
        --model $model \
        --output ${model}_model.joblib
done
```

### Ejemplo: Pipeline Programático (Python)

```python
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.train import train_model
from mlops_project.modeling.predict import predict_and_evaluate

# 1. Preparar datos
X_train, X_test, y_train, y_test = load_and_prepare_data(
    filepath="data/raw/german_credit_modified.csv",
    save_processed=True
)

# 2. Preparar features
X_train_t, X_test_t, preprocessor = prepare_features(
    X_train=X_train,
    X_test=X_test,
    save_preprocessor=True
)

# 3. Entrenar modelo
pipeline, results = train_model(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor,
    model_name="logistic_regression",
    use_smote=True,
    evaluate=True,
    save_model=True
)

# 4. Evaluar
y_pred, y_proba, metrics = predict_and_evaluate(
    model=pipeline,
    X_test=X_test,
    y_test=y_test,
    save_predictions=True
)

print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"F1-Score: {metrics['f1']:.4f}")
```

## Testing

### Ejecutar Tests

```bash
# Todos los tests (usando UV)
make test

# Tests con coverage
make test-cov

# Ver reporte de coverage
open htmlcov/index.html

# Tests específicos con UV (sin activar venv)
uv run pytest tests/test_config.py -v
uv run pytest tests/test_modeling.py::TestModelInstances -v
```

### Estadísticas de Testing

- **Total de Tests**: 130
- **Tests Pasando**: 108 (83%)
- **Coverage**: 64%
- **Uso de Mocks**: Para I/O, entrenamiento y visualización

## Configuración

Toda la configuración del proyecto está centralizada en `mlops_project/config.py`:

```python
# Ejemplo de uso de configuración
from mlops_project.config import (
    RANDOM_SEED,
    NUMERIC_FEATURES,
    NOMINAL_FEATURES,
    BEST_MODEL_PARAMS,
    get_data_path,
    get_model_path
)

# Obtener rutas
data_path = get_data_path("train.csv", "processed")
model_path = get_model_path("model.joblib")

# Usar parámetros
print(f"Semilla aleatoria: {RANDOM_SEED}")
print(f"Features numéricas: {NUMERIC_FEATURES}")
```

### Parámetros Configurables

- **Semilla aleatoria**: `RANDOM_SEED = 42`
- **División de datos**: `TEST_SIZE = 0.3`
- **Cross-validation**: `CV_FOLDS = 5`, `CV_REPEATS = 3`
- **SMOTE**: `method = "BorderlineSMOTE"`
- **Features**: Listas de features numéricas, ordinales y nominales
- **Modelos**: Hiperparámetros para cada modelo disponible

## Comandos Make Útiles

### Gestión del Proyecto

```bash
make help              # Mostrar todos los comandos
make init              # Inicializar proyecto
make requirements      # Instalar dependencias
make clean             # Limpiar archivos temporales
```

### Calidad de Código

```bash
make lint              # Verificar código con ruff (usa UV internamente)
make format            # Formatear código (usa UV internamente)
make check             # Lint + tests (todo con UV)
```

### Pipeline

```bash
make prepare-data      # Preparar datos
make prepare-features  # Preparar features
make train             # Entrenar modelo
make predict           # Realizar predicciones
make pipeline          # Pipeline completo
```

### Datos (S3)

```bash
make sync-data-down    # Descargar datos desde S3
make sync-data-up      # Subir datos a S3
```

### Utilidades

```bash
make tree              # Mostrar estructura del proyecto
make scripts           # Listar scripts disponibles
make python-version    # Mostrar versión de Python
```

## Modelos Disponibles

El proyecto soporta múltiples modelos de clasificación:

1. **Logistic Regression** (Recomendado)
   - Mejor performance en validación cruzada
   - ROC-AUC: ~0.77
   - Interpretable y rápido

2. **Random Forest**
   - Ensemble method robusto
   - Buena capacidad de generalización

3. **Decision Tree**
   - Modelo simple e interpretable
   - Útil para análisis exploratorio

4. **Support Vector Machine (SVM)**
   - Kernel RBF
   - Bueno para clasificación no lineal

5. **XGBoost**
   - Gradient boosting
   - Alta performance en muchos casos

## Próximos Pasos

### Integración con MLFlow

El proyecto está preparado para integración con MLFlow:

```python
# Ejemplo de integración futura
import mlflow

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(BEST_MODEL_PARAMS)
    
    # Train model
    pipeline, results = train_model(...)
    
    # Log metrics
    mlflow.log_metrics(results['accuracy'])
    
    # Log model
    mlflow.sklearn.log_model(pipeline, "model")
```

### Orquestación con Airflow

Para automatización completa, se puede integrar con Apache Airflow:

```python
# Ejemplo de DAG de Airflow (futuro)
from airflow import DAG
from airflow.operators.bash import BashOperator

dag = DAG('mlops_pipeline', schedule_interval='@daily')

prepare_data = BashOperator(
    task_id='prepare_data',
    bash_command='uv run mlops-prepare-data --input data/raw/data.csv --save',
    dag=dag
)

prepare_features = BashOperator(
    task_id='prepare_features',
    bash_command='uv run mlops-prepare-features --train data/processed/X.csv --save-preprocessor',
    dag=dag
)

train_model = BashOperator(
    task_id='train_model',
    bash_command='uv run mlops-train ...',
    dag=dag
)

prepare_data >> prepare_features >> train_model
```

### Deployment

- **FastAPI**: Crear API REST para serving
- **Docker**: Containerización del pipeline
- **CI/CD**: GitHub Actions para automatización
- **Monitoring**: Integrar con Prometheus/Grafana

Ver `mlops_project/README.md` para más detalles sobre orquestación.

## Convenciones

- **Código**: Inglés
- **Comentarios**: Español
- **Documentación**: Español
- **Commits**: Español (convención del equipo)
- **Gestión de Paquetes**: UV (no pip directamente)
- **Ejecución**: `uv run` para todos los comandos Python (no requiere activar venv)

## Equipo

**Equipo 29** - TC5044.10  
Maestría en Ciencia de Datos - ITESM

## Licencia

BSD License - Ver archivo `LICENSE` para más detalles.

---

**Fecha**: Octubre 2025  
**Versión**: 0.0.1  
**Python**: 3.12.0
