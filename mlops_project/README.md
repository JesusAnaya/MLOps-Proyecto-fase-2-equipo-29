# Guía de Orquestación del Pipeline MLOps

Este documento proporciona ejemplos y estrategias para orquestar el pipeline MLOps de manera automatizada, preparado para escalar hacia soluciones empresariales.

## Tabla de Contenidos

- [Introducción](#introducción)
- [Arquitectura del Pipeline](#arquitectura-del-pipeline)
- [Orquestación Manual](#orquestación-manual)
- [Orquestación con Scripts](#orquestación-con-scripts)
- [Orquestación Programática](#orquestación-programática)
- [Integración con Orquestadores](#integración-con-orquestadores)
- [Monitoreo y Logging](#monitoreo-y-logging)
- [Mejores Prácticas](#mejores-prácticas)

## Introducción

El pipeline MLOps implementado en este proyecto sigue una arquitectura modular que facilita la automatización y orquestación. Cada componente puede ejecutarse de forma independiente o como parte de un flujo completo.

### Componentes del Pipeline

1. **Preparación de Datos** (`dataset.py`)
2. **Ingeniería de Features** (`features.py`)
3. **Entrenamiento de Modelos** (`modeling/train.py`)
4. **Predicción e Inferencia** (`modeling/predict.py`)

## Arquitectura del Pipeline

```
┌─────────────────────┐
│  Datos Crudos (Raw) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  dataset.py         │ ◄── Limpieza, validación, división
│  - DataLoader       │
│  - DataCleaner      │
│  - DataSplitter     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Datos Limpios      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  features.py        │ ◄── Transformación, encoding, escalado
│  - InvalidDataHandler│
│  - OutlierHandler   │
│  - FeaturePreprocessor│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Features Procesadas│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  modeling/train.py  │ ◄── Entrenamiento, validación, guardado
│  - Cross-validation │
│  - SMOTE (opcional) │
│  - Model selection  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Modelo Entrenado   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  modeling/predict.py│ ◄── Inferencia, evaluación
│  - Predicciones     │
│  - Métricas         │
└─────────────────────┘
```

## Orquestación Manual

### Ejecución Paso a Paso

```bash
# Paso 1: Preparar datos
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save

# Paso 2: Preparar features
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor

# Paso 3: Entrenar modelo
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model logistic_regression

# Paso 4: Realizar predicciones
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save
```

### Con Makefile

```bash
# Pipeline completo
make pipeline

# O paso a paso
make prepare-data
make prepare-features
make train
make predict
```

## Orquestación con Scripts

### Script Bash Básico

Crear `scripts/run_pipeline.sh`:

```bash
#!/bin/bash

# Script de orquestación del pipeline MLOps
# Uso: ./scripts/run_pipeline.sh [modelo]

set -e  # Salir si hay error

MODEL=${1:-logistic_regression}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "=== Pipeline MLOps - Inicio: $TIMESTAMP ==="
echo "Modelo seleccionado: $MODEL"

# 1. Preparar datos
echo "[1/4] Preparando datos..."
uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save \
    2>&1 | tee $LOG_DIR/prepare_data_$TIMESTAMP.log

# 2. Preparar features
echo "[2/4] Preparando features..."
uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor \
    2>&1 | tee $LOG_DIR/prepare_features_$TIMESTAMP.log

# 3. Entrenar modelo
echo "[3/4] Entrenando modelo $MODEL..."
uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model $MODEL \
    2>&1 | tee $LOG_DIR/train_$TIMESTAMP.log

# 4. Evaluar modelo
echo "[4/4] Evaluando modelo..."
uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save \
    2>&1 | tee $LOG_DIR/predict_$TIMESTAMP.log

echo "=== Pipeline completado exitosamente ==="
echo "Logs guardados en: $LOG_DIR/"
```

### Script Bash con Manejo de Errores

Crear `scripts/run_pipeline_robust.sh`:

```bash
#!/bin/bash

# Pipeline robusto con manejo de errores y notificaciones

set -euo pipefail  # Modo estricto

# Configuración
MODEL=${1:-logistic_regression}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/${TIMESTAMP}"
mkdir -p $LOG_DIR

# Funciones auxiliares
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_DIR/pipeline.log
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a $LOG_DIR/pipeline.log
}

cleanup() {
    log_info "Limpiando archivos temporales..."
    # Agregar lógica de limpieza si es necesario
}

trap cleanup EXIT

# Inicio del pipeline
log_info "=== Pipeline MLOps Iniciado ==="
log_info "Modelo: $MODEL"
log_info "Timestamp: $TIMESTAMP"

# Validar que existan los datos de entrada
if [ ! -f "data/raw/german_credit_modified.csv" ]; then
    log_error "Archivo de datos no encontrado"
    exit 1
fi

# 1. Preparar datos
log_info "[1/4] Preparando datos..."
if uv run mlops-prepare-data \
    --input data/raw/german_credit_modified.csv \
    --save \
    > $LOG_DIR/step1_prepare_data.log 2>&1; then
    log_info "✓ Preparación de datos completada"
else
    log_error "✗ Error en preparación de datos"
    exit 1
fi

# 2. Preparar features
log_info "[2/4] Preparando features..."
if uv run mlops-prepare-features \
    --train data/processed/Xtraintest.csv \
    --save-preprocessor \
    > $LOG_DIR/step2_prepare_features.log 2>&1; then
    log_info "✓ Preparación de features completada"
else
    log_error "✗ Error en preparación de features"
    exit 1
fi

# 3. Entrenar modelo
log_info "[3/4] Entrenando modelo..."
if uv run mlops-train \
    --X-train data/processed/Xtraintest.csv \
    --y-train data/processed/ytraintest.csv \
    --preprocessor models/preprocessor.joblib \
    --model $MODEL \
    > $LOG_DIR/step3_train.log 2>&1; then
    log_info "✓ Entrenamiento completado"
else
    log_error "✗ Error en entrenamiento"
    exit 1
fi

# 4. Evaluar
log_info "[4/4] Evaluando modelo..."
if uv run mlops-predict \
    --model models/best_model.joblib \
    --X-test data/processed/Xtraintest.csv \
    --y-test data/processed/ytraintest.csv \
    --save \
    > $LOG_DIR/step4_predict.log 2>&1; then
    log_info "✓ Evaluación completada"
else
    log_error "✗ Error en evaluación"
    exit 1
fi

log_info "=== Pipeline completado exitosamente ==="
log_info "Resultados en: $LOG_DIR/"
```

## Orquestación Programática

### Pipeline Python Básico

Crear `scripts/run_pipeline.py`:

```python
#!/usr/bin/env python
"""
Script de orquestación programática del pipeline MLOps.

Uso:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --model random_forest
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from mlops_project.config import get_data_path, get_model_path
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.predict import predict_and_evaluate
from mlops_project.modeling.train import train_model

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_pipeline(model_name: str = "logistic_regression", use_smote: bool = True):
    """
    Ejecuta el pipeline MLOps completo.

    Args:
        model_name: Nombre del modelo a entrenar
        use_smote: Si se debe usar SMOTE para balanceo
    """
    logger.info("=" * 60)
    logger.info("INICIANDO PIPELINE MLOPS")
    logger.info("=" * 60)
    logger.info(f"Modelo: {model_name}")
    logger.info(f"SMOTE: {'Sí' if use_smote else 'No'}")

    try:
        # 1. Preparar datos
        logger.info("[1/4] Preparando datos...")
        raw_data_path = get_data_path("german_credit_modified.csv", "raw")

        X_train, X_test, y_train, y_test = load_and_prepare_data(
            filepath=raw_data_path, save_processed=True, return_combined=False
        )
        logger.info(f"✓ Datos preparados: {len(X_train) + len(X_test)} muestras")

        # 2. Preparar features
        logger.info("[2/4] Preparando features...")
        X_train_t, X_test_t, preprocessor = prepare_features(
            X_train=X_train, X_test=X_test, save_preprocessor=True
        )
        logger.info(f"✓ Features preparadas: {X_train_t.shape[1]} features")

        # 3. Entrenar modelo
        logger.info(f"[3/4] Entrenando modelo {model_name}...")
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name=model_name,
            use_smote=use_smote,
            evaluate=True,
            save_model=True,
        )

        if results:
            roc_auc = results["roc_auc"]["test_mean"]
            logger.info(f"✓ Modelo entrenado. ROC-AUC: {roc_auc:.4f}")

        # 4. Evaluar
        logger.info("[4/4] Evaluando modelo...")
        y_pred, y_proba, metrics = predict_and_evaluate(
            model=pipeline, X_test=X_test, y_test=y_test, save_predictions=True
        )

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        return metrics

    except Exception as e:
        logger.error(f"Error en el pipeline: {e}")
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Orquestación del pipeline MLOps")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "decision_tree", "svm", "xgboost"],
        help="Modelo a entrenar",
    )
    parser.add_argument("--no-smote", action="store_true", help="No usar SMOTE")

    args = parser.parse_args()

    # Crear directorio de logs
    Path("logs").mkdir(exist_ok=True)

    # Ejecutar pipeline
    metrics = run_pipeline(model_name=args.model, use_smote=not args.no_smote)

    return 0


if __name__ == "__main__":
    exit(main())
```

### Pipeline con Experimentación

Crear `scripts/experiment_pipeline.py`:

```python
#!/usr/bin/env python
"""
Script para experimentar con múltiples configuraciones del pipeline.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from mlops_project.config import AVAILABLE_MODELS, get_data_path
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.train import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_experiment(experiment_config):
    """
    Ejecuta un experimento con configuración específica.

    Args:
        experiment_config: Diccionario con configuración del experimento
    """
    logger.info(f"Ejecutando experimento: {experiment_config['name']}")

    # Preparar datos (una sola vez)
    raw_data_path = get_data_path("german_credit_modified.csv", "raw")
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        filepath=raw_data_path, save_processed=False, return_combined=False
    )

    # Preparar features
    X_train_t, X_test_t, preprocessor = prepare_features(
        X_train=X_train, X_test=X_test, save_preprocessor=False
    )

    # Entrenar modelo
    pipeline, results = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        model_name=experiment_config["model"],
        use_smote=experiment_config["use_smote"],
        smote_method=experiment_config.get("smote_method", "BorderlineSMOTE"),
        evaluate=True,
        save_model=False,
    )

    return results


def main():
    """Ejecuta múltiples experimentos."""
    # Definir experimentos
    experiments = [
        {"name": "LogReg + SMOTE", "model": "logistic_regression", "use_smote": True},
        {
            "name": "LogReg sin SMOTE",
            "model": "logistic_regression",
            "use_smote": False,
        },
        {
            "name": "RandomForest + BorderlineSMOTE",
            "model": "random_forest",
            "use_smote": True,
            "smote_method": "BorderlineSMOTE",
        },
        {"name": "XGBoost + SMOTE", "model": "xgboost", "use_smote": True},
        {"name": "SVM + SMOTE", "model": "svm", "use_smote": True},
    ]

    # Ejecutar experimentos
    results_all = {}
    for exp_config in experiments:
        logger.info("=" * 60)
        try:
            results = run_experiment(exp_config)
            results_all[exp_config["name"]] = {
                "config": exp_config,
                "metrics": results,
            }
            logger.info(f"✓ {exp_config['name']} completado")
        except Exception as e:
            logger.error(f"✗ Error en {exp_config['name']}: {e}")
            results_all[exp_config["name"]] = {"config": exp_config, "error": str(e)}

    # Guardar resultados
    output_dir = Path("experiments") / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_all, f, indent=2)

    logger.info(f"Resultados guardados en: {output_dir}")

    # Mostrar comparación
    logger.info("\n" + "=" * 60)
    logger.info("COMPARACIÓN DE EXPERIMENTOS")
    logger.info("=" * 60)

    for name, data in results_all.items():
        if "metrics" in data:
            roc_auc = data["metrics"]["roc_auc"]["test_mean"]
            logger.info(f"{name:30s} | ROC-AUC: {roc_auc:.4f}")
        else:
            logger.info(f"{name:30s} | ERROR")


if __name__ == "__main__":
    main()
```

## Integración con Orquestadores

### Preparación para Apache Airflow

```python
"""
Ejemplo de DAG de Airflow para el pipeline MLOps.

NOTA: Este es un ejemplo de cómo se vería la integración con Airflow.
No ejecutar directamente - requiere instalación y configuración de Airflow.

Instalación de Airflow (futuro):
    uv add apache-airflow
    uv run airflow db init
    uv run airflow webserver --port 8080
    uv run airflow scheduler
"""

from datetime import datetime, timedelta

# Imports de Airflow (comentados para no requerir instalación)
# from airflow import DAG
# from airflow.operators.bash import BashOperator
# from airflow.operators.python import PythonOperator
# from airflow.sensors.filesystem import FileSensor

# Configuración del DAG
default_args = {
    "owner": "equipo29",
    "depends_on_past": False,
    "email": ["equipo29@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Definición del DAG
dag = DAG(
    "mlops_credit_pipeline",
    default_args=default_args,
    description="Pipeline MLOps para clasificación de crédito",
    schedule_interval="@weekly",  # Ejecutar semanalmente
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["mlops", "credit", "classification"],
)

# Tarea 1: Verificar que existan datos nuevos
check_data = FileSensor(
    task_id="check_new_data",
    filepath="data/raw/german_credit_modified.csv",
    poke_interval=300,  # 5 minutos
    timeout=3600,  # 1 hora
    dag=dag,
)

# Tarea 2: Preparar datos
prepare_data = BashOperator(
    task_id="prepare_data",
    bash_command="cd /path/to/project && uv run mlops-prepare-data --input data/raw/german_credit_modified.csv --save",
    dag=dag,
)

# Tarea 3: Preparar features
prepare_features = BashOperator(
    task_id="prepare_features",
    bash_command="cd /path/to/project && uv run mlops-prepare-features --train data/processed/Xtraintest.csv --save-preprocessor",
    dag=dag,
)

# Tarea 4: Entrenar modelo
train_model = BashOperator(
    task_id="train_model",
    bash_command="cd /path/to/project && uv run mlops-train --X-train data/processed/Xtraintest.csv --y-train data/processed/ytraintest.csv --preprocessor models/preprocessor.joblib --model logistic_regression",
    dag=dag,
)

# Tarea 5: Evaluar modelo
evaluate_model = BashOperator(
    task_id="evaluate_model",
    bash_command="cd /path/to/project && uv run mlops-predict --model models/best_model.joblib --X-test data/processed/Xtraintest.csv --y-test data/processed/ytraintest.csv --save",
    dag=dag,
)

# Definir dependencias
check_data >> prepare_data >> prepare_features >> train_model >> evaluate_model
```

### Preparación para MLflow

```python
"""
Ejemplo de integración con MLflow para tracking de experimentos.

NOTA: Este código muestra cómo se integraría MLflow en el futuro.
No ejecutar directamente - requiere instalación y configuración de MLflow.

Instalación de MLflow (futuro):
    uv add mlflow
    uv run mlflow ui --port 5000
"""

import mlflow
import mlflow.sklearn

from mlops_project.config import BEST_MODEL_PARAMS, RANDOM_SEED
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
from mlops_project.modeling.train import train_model


def run_experiment_with_mlflow(model_name: str, use_smote: bool):
    """
    Ejecuta experimento con tracking en MLflow.

    Args:
        model_name: Nombre del modelo
        use_smote: Si usar SMOTE
    """
    # Iniciar run de MLflow
    with mlflow.start_run(run_name=f"{model_name}_smote-{use_smote}"):
        # Log de parámetros
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_param("random_seed", RANDOM_SEED)
        mlflow.log_params(BEST_MODEL_PARAMS)

        # Preparar datos
        X_train, X_test, y_train, y_test = load_and_prepare_data(
            "data/raw/german_credit_modified.csv"
        )

        # Preparar features
        X_train_t, X_test_t, preprocessor = prepare_features(X_train, X_test)

        # Entrenar modelo
        pipeline, results = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            model_name=model_name,
            use_smote=use_smote,
            evaluate=True,
        )

        # Log de métricas
        if results:
            for metric_name, scores in results.items():
                mlflow.log_metric(f"{metric_name}_test_mean", scores["test_mean"])
                mlflow.log_metric(f"{metric_name}_test_std", scores["test_std"])
                mlflow.log_metric(f"{metric_name}_train_mean", scores["train_mean"])
                mlflow.log_metric(f"{metric_name}_train_std", scores["train_std"])

        # Log del modelo
        mlflow.sklearn.log_model(pipeline, "model")

        # Log de artefactos
        mlflow.log_artifact("models/model_results.json")

        print(f"Run ID: {mlflow.active_run().info.run_id}")


# Ejemplo de uso
if __name__ == "__main__":
    # Configurar MLflow
    mlflow.set_experiment("credit_classification")
    mlflow.set_tracking_uri("http://localhost:5000")

    # Ejecutar experimentos
    models = ["logistic_regression", "random_forest", "xgboost"]
    for model in models:
        for use_smote in [True, False]:
            run_experiment_with_mlflow(model, use_smote)
```

## Monitoreo y Logging

### Configuración de Logging

```python
"""
Configuración centralizada de logging para el proyecto.
"""

import logging
from pathlib import Path


def setup_logging(log_level=logging.INFO):
    """Configura logging para el proyecto."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "mlops_pipeline.log"),
            logging.StreamHandler(),
        ],
    )
```

## Mejores Prácticas

### 1. Reproducibilidad

```python
# Siempre usar la semilla aleatoria del config
from mlops_project.config import RANDOM_SEED
import numpy as np
import random

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### 2. Versionado de Datos y Modelos

```bash
# Usar timestamps en nombres de archivos
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
uv run mlops-train --output models/model_$TIMESTAMP.joblib
```

### 3. Validación de Entrada

```python
# Validar que existan archivos antes de procesar
from pathlib import Path

def validate_inputs():
    required_files = [
        "data/raw/german_credit_modified.csv",
        "models/preprocessor.joblib",
    ]
    for file in required_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Archivo requerido no encontrado: {file}")
```

### 4. Manejo de Errores

```python
# Usar try-except para manejo robusto
try:
    X_train, X_test, y_train, y_test = load_and_prepare_data(...)
except Exception as e:
    logger.error(f"Error en preparación de datos: {e}")
    # Notificar, guardar estado, etc.
    raise
```

### 5. Testing del Pipeline

```bash
# Ejecutar tests antes de pipeline en producción
make test
make check
```

## Próximos Pasos

### Escalamiento Futuro

1. **Containerización con Docker**:
   - Crear Dockerfile
   - Docker Compose para servicios
   - Kubernetes para orquestación a escala

2. **CI/CD con GitHub Actions**:
   - Tests automáticos en PRs
   - Deployment automático
   - Versionado de modelos

3. **API de Serving con FastAPI**:
   - Endpoint de predicción REST
   - Documentación automática con Swagger
   - Autenticación y rate limiting

4. **Monitoreo en Producción**:
   - Prometheus para métricas
   - Grafana para dashboards
   - Alertas automáticas

5. **Data Drift Detection**:
   - Monitorear cambios en distribución de datos
   - Re-entrenamiento automático cuando sea necesario

## Recursos Adicionales

- **Documentación de UV**: https://docs.astral.sh/uv/
- **MLflow Docs**: https://mlflow.org/docs/latest/index.html
- **Apache Airflow**: https://airflow.apache.org/docs/
- **FastAPI**: https://fastapi.tiangolo.com/

---

**Nota**: Este documento está en constante evolución. Contribuciones y sugerencias son bienvenidas.

**Equipo 29** - TC5044.10

