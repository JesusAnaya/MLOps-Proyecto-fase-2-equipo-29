#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mlops-proyecto-etapa-2-equipo-29
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMANDOS DE GESTIÓN                                                          #
#################################################################################

## Instalar dependencias del proyecto
.PHONY: requirements
requirements:
	uv sync
	@echo ">>> Dependencias instaladas con UV"

## Instalar dependencias de desarrollo
.PHONY: requirements-dev
requirements-dev:
	uv sync --dev
	@echo ">>> Dependencias de desarrollo instaladas"

## Limpiar archivos temporales de Python
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	@echo ">>> Archivos temporales eliminados"

#################################################################################
# COMANDOS DE CALIDAD DE CÓDIGO                                                #
#################################################################################

## Verificar código con ruff
.PHONY: lint
lint:
	uv run ruff format --check
	uv run ruff check
	@echo ">>> Verificación de código completada"

## Formatear código con ruff
.PHONY: format
format:
	uv run ruff check --fix
	uv run ruff format
	@echo ">>> Código formateado"

## Ejecutar todos los tests
.PHONY: test
test:
	uv run pytest
	@echo ">>> Tests ejecutados"

## Ejecutar tests con reporte de coverage
.PHONY: test-cov
test-cov:
	uv run pytest --cov=mlops_project --cov-report=html --cov-report=term
	@echo ">>> Tests con coverage completados"
	@echo ">>> Ver reporte en htmlcov/index.html"

## Ejecutar verificación completa (lint + tests)
.PHONY: check
check: lint test
	@echo ">>> Verificación completa de calidad completada"

#################################################################################
# PIPELINE MLOps                                                                #
#################################################################################

## Preparar datos (carga y limpieza)
.PHONY: prepare-data
prepare-data:
	uv run mlops-prepare-data \
		--input data/raw/german_credit_modified.csv \
		--save
	@echo ">>> Datos preparados"

## Preparar features (transformación)
.PHONY: prepare-features
prepare-features:
	uv run mlops-prepare-features \
		--train data/processed/Xtraintest.csv \
		--save-preprocessor
	@echo ">>> Features preparadas"

## Entrenar modelo
.PHONY: train
train:
	uv run mlops-train \
		--X-train data/processed/Xtraintest.csv \
		--y-train data/processed/ytraintest.csv \
		--preprocessor models/preprocessor.joblib \
		--model logistic_regression
	@echo ">>> Modelo entrenado"

## Realizar predicciones
.PHONY: predict
predict:
	uv run mlops-predict \
		--model models/best_model.joblib \
		--X-test data/processed/Xtraintest.csv \
		--y-test data/processed/ytraintest.csv \
		--save
	@echo ">>> Predicciones realizadas"

## Ejecutar pipeline completo
.PHONY: pipeline
pipeline: prepare-data prepare-features train
	@echo ">>> Pipeline MLOps completado exitosamente"

#################################################################################
# GESTIÓN DE DATOS (DVC + S3)                                                  #
#################################################################################

## Descargar datos desde S3 con DVC
.PHONY: dvc-pull
dvc-pull:
	dvc pull
	@echo ">>> Datos descargados desde S3 con DVC"

## Subir datos a S3 con DVC
.PHONY: dvc-push
dvc-push:
	dvc push
	@echo ">>> Datos subidos a S3 con DVC"

## Ver estado de DVC
.PHONY: dvc-status
dvc-status:
	dvc status
	@echo ">>> Estado de DVC mostrado"

## Agregar nuevo archivo a DVC
.PHONY: dvc-add
dvc-add:
	@echo "Uso: dvc add <archivo>"
	@echo "Ejemplo: dvc add data/raw/nuevo_dataset.csv"
	@echo "Después: git add <archivo>.dvc .gitignore"

#################################################################################
# GESTIÓN DEL ENTORNO                                                          #
#################################################################################

## Crear entorno virtual con UV
.PHONY: create-environment
create-environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Entorno virtual UV creado"
	@echo ">>> Nota: UV permite ejecutar comandos sin activar el venv"
	@echo ">>>   Usar: uv run <comando>"
	@echo ">>>   Ejemplo: uv run pytest tests/"
	@echo ""
	@echo ">>> Opcional - Activar manualmente:"
	@echo ">>>   Unix/macOS: source .venv/bin/activate"
	@echo ">>>   Windows: .venv\\Scripts\\activate"

## Inicializar proyecto (crear entorno e instalar dependencias)
.PHONY: init
init: create-environment requirements
	@echo ">>> Proyecto inicializado"

#################################################################################
# UTILIDADES                                                                    #
#################################################################################

## Mostrar estructura del proyecto
.PHONY: tree
tree:
	tree -I '__pycache__|*.pyc|.git|.venv|*.egg-info|.pytest_cache|htmlcov' -L 3

## Listar scripts disponibles
.PHONY: scripts
scripts:
	@echo "=== Scripts Disponibles ==="
	@echo ""
	@echo "Preparación de Datos:"
	@echo "  mlops-prepare-data      - Cargar y limpiar datos"
	@echo "  mlops-prepare-features  - Transformar features"
	@echo ""
	@echo "Modelado:"
	@echo "  mlops-train            - Entrenar modelo"
	@echo "  mlops-predict          - Realizar predicciones"
	@echo ""
	@echo "Uso: uv run <script-name> [argumentos]"
	@echo ""
	@echo "Ejemplos:"
	@echo "  uv run mlops-prepare-data --input data/raw/data.csv --save"
	@echo "  uv run mlops-train --model logistic_regression"
	@echo ""

## Mostrar versión de Python
.PHONY: python-version
python-version:
	@uv run python --version

#################################################################################
# AYUDA                                                                         #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Comandos Disponibles:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
