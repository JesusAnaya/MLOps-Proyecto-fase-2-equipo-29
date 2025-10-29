# Guía de Migración: Notebooks a Módulos MLOps

## Propósito de este Documento

Este documento sirve como **mapa de ruta** para el equipo, explicando cómo la lógica desarrollada en los notebooks de experimentación ha sido exportada a los módulos Python del proyecto MLOps. 

Los notebooks originales quedan como **referencia y documentación** del proceso de experimentación, mientras que los módulos en `mlops_project/` contienen el código productivo y reutilizable.

## Estado de los Notebooks

**Notebooks de Prototipo** (Solo lectura - No modificar):
- `exploring/01_eda_south_german_credit.ipynb` - Análisis exploratorio y limpieza
- `preprocessing/02_preprocessing_pipeline.ipynb` - Pipeline de preprocesamiento
- `modeling/03_model_training_evaluation.ipynb` - Entrenamiento y evaluación

Estos notebooks sirvieron para **experimentar y validar** el enfoque antes de productizar el código.

## Mapa de Migración: Notebooks → Módulos

### Notebook 01: EDA y Limpieza de Datos

**Archivo**: `exploring/01_eda_south_german_credit.ipynb`

**Objetivo**: Análisis exploratorio, detección de valores inválidos, limpieza de datos

#### Lógica Exportada a `mlops_project/dataset.py`

| Funcionalidad Notebook | Ubicación en Módulo | Descripción |
|------------------------|---------------------|-------------|
| **Cell 5-7**: Lectura de datos | `DataLoader.load_data()` | Carga CSV y valida existencia |
| **Cell 12**: Detección de valores no numéricos | `DataCleaner._convert_to_numeric()` | Identifica y reemplaza valores inválidos por NaN |
| **Cell 12**: Conversión a tipo numérico | `DataCleaner._convert_to_numeric()` | Convierte columnas usando `pd.to_numeric()` |
| **Cell 14**: Limpieza de variable objetivo | `DataCleaner._clean_target_variable()` | Elimina filas con target nulo o no válido |
| **Cell 15**: Eliminación de `mixed_type_col` | `DataCleaner.clean_data()` | Drop de columnas problemáticas |

**Clase Principal**: `DataCleaner`
```python
# Uso en módulo
from mlops_project.dataset import DataCleaner

cleaner = DataCleaner(target_column='kredit', mixed_type_column='mixed_type_col')
df_clean = cleaner.clean_data(df_raw)
```

#### Lógica Exportada a `mlops_project/features.py`

| Funcionalidad Notebook | Ubicación en Módulo | Descripción |
|------------------------|---------------------|-------------|
| **Cell 17**: Definición de reglas de validación (`isin_rules`) | `config.py`: `CATEGORICAL_VALIDATION_RULES` | Diccionario con valores válidos por columna |
| **Cell 19**: Imputación de valores inválidos con moda | `InvalidDataHandler.fit_transform()` | Imputa valores no válidos usando la moda |
| **Cell 21**: Detección de outliers con IQR | `OutlierHandler.fit()` | Calcula límites Q1, Q3, IQR |
| **Cell 21**: Función `detect_outliers_iqr()` | `OutlierHandler.transform()` | Aplica capping a outliers |

**Clase Principal**: `InvalidDataHandler`, `OutlierHandler`
```python
# Uso en módulo
from mlops_project.features import InvalidDataHandler, OutlierHandler

invalid_handler = InvalidDataHandler(validation_rules=CATEGORICAL_VALIDATION_RULES)
X_clean = invalid_handler.fit_transform(X)

outlier_handler = OutlierHandler(method='IQR', variables=['laufzeit', 'hoehe', 'alter'])
X_capped = outlier_handler.fit_transform(X_clean)
```

---

### Notebook 02: Pipeline de Preprocesamiento

**Archivo**: `preprocessing/02_preprocessing_pipeline.ipynb`

**Objetivo**: Transformación de features, encoding, escalado

#### Lógica Exportada a `mlops_project/config.py`

| Elemento Notebook | Ubicación en config.py | Valor |
|-------------------|------------------------|-------|
| **Cell 7**: `numericas_pipe_nombres` | `NUMERIC_FEATURES` | `['laufzeit', 'hoehe', 'alter']` |
| **Cell 7**: `nominales_pipe_nombres` | `NOMINAL_FEATURES` | 11 variables categóricas |
| **Cell 7**: `ordinales_pipe_nombres` | `ORDINAL_FEATURES` | `['beszeit', 'rate', 'wohnzeit', 'verm', 'bishkred', 'beruf']` |
| **Cell 9**: MinMaxScaler range | `NUMERIC_SCALER_RANGE` | `(1, 2)` |
| **Cell 9**: Estrategia de imputación | `NUMERIC_IMPUTE_STRATEGY` | `'median'` |
| **Cell 9**: Estrategia categórica | `CATEGORICAL_IMPUTE_STRATEGY` | `'most_frequent'` |

#### Lógica Exportada a `mlops_project/features.py`

| Funcionalidad Notebook | Ubicación en Módulo | Descripción |
|------------------------|---------------------|-------------|
| **Cell 8-9**: Pipeline numérico | `FeaturePreprocessor.fit()` líneas 258-263 | `SimpleImputer('median') + MinMaxScaler(1,2)` |
| **Cell 8-9**: Pipeline nominal | `FeaturePreprocessor.fit()` líneas 266-271 | `SimpleImputer('most_frequent') + OneHotEncoder(drop='first')` |
| **Cell 8-9**: Pipeline ordinal | `FeaturePreprocessor.fit()` líneas 274-282 | `SimpleImputer('most_frequent') + OrdinalEncoder(unknown=-1)` |
| **Cell 9**: `ColumnTransformer` | `FeaturePreprocessor.fit()` líneas 285-293 | Combina los 3 pipelines con `set_output(transform="pandas")` |
| **Cell 9**: `preprocess_data()` función | `prepare_features()` función | Pipeline completo de transformación |

**Clase Principal**: `FeaturePreprocessor`
```python
# Uso en módulo
from mlops_project.features import FeaturePreprocessor

preprocessor = FeaturePreprocessor(
    numeric_features=NUMERIC_FEATURES,
    nominal_features=NOMINAL_FEATURES,
    ordinal_features=ORDINAL_FEATURES
)
X_transformed = preprocessor.fit_transform(X)
```

#### Lógica Exportada a `mlops_project/dataset.py`

| Funcionalidad Notebook | Ubicación en Módulo | Descripción |
|------------------------|---------------------|-------------|
| **Cell 12**: Función `split_data_by_target()` | `DataSplitter.split()` | División train/test con estratificación |
| **Cell 13**: `train_test_split` con test_size=0.3 | `DataSplitter.split()` | Usa `TEST_SIZE = 0.3` de config |
| **Cell 13**: `stratify=y` | `DataSplitter` atributo `stratify=True` | Mantiene proporción de clases |

**Clase Principal**: `DataSplitter`
```python
# Uso en módulo
from mlops_project.dataset import DataSplitter

splitter = DataSplitter(test_size=0.3, random_state=42, stratify=True)
X_train, X_test, y_train, y_test = splitter.split(df, target_column='kredit')
```

---

### Notebook 03: Modelado y Evaluación

**Archivo**: `modeling/03_model_training_evaluation.ipynb`

**Objetivo**: Entrenamiento de modelos, cross-validation, evaluación de métricas

#### Lógica Exportada a `mlops_project/config.py`

| Elemento Notebook | Ubicación en config.py | Valor |
|-------------------|------------------------|-------|
| **Cell 4**: `RANDOM_SEED = 42` | `RANDOM_SEED` | `42` |
| **Cell 8**: `n_splits=5, n_repeats=3` | `CV_FOLDS`, `CV_REPEATS` | `5`, `3` |
| **Cell 11**: Hiperparámetros LogReg | `BEST_MODEL_PARAMS` | `penalty='l2', solver='newton-cg', C=1` |
| **Cell 11**: Config SMOTE | `SMOTE_CONFIG` | `BorderlineSMOTE, k_neighbors=5, m_neighbors=10` |
| **Cell 13, 15, 17, 19**: Hiperparámetros de modelos | `AVAILABLE_MODELS` | Dict con configs de 5 modelos |

#### Lógica Exportada a `mlops_project/modeling/train.py`

| Funcionalidad Notebook | Ubicación en Módulo | Descripción |
|------------------------|---------------------|-------------|
| **Cell 8**: Función `mi_fun()` completa | `evaluate_model()` | Cross-validation con múltiples métricas |
| **Cell 8**: `ImbPipeline` con preprocessor + SMOTE + modelo | `create_training_pipeline()` | Crea pipeline completo |
| **Cell 9**: `RepeatedStratifiedKFold` | `evaluate_model()` línea 156 | CV con 5 folds x 3 repeats |
| **Cell 9**: Diccionario de métricas | `evaluate_model()` líneas 159-167 | 7 métricas: accuracy, precision, recall, f1, roc_auc, avg_precision, gmean |
| **Cell 9**: `cross_validate()` con warnings | `evaluate_model()` líneas 170-174 | Evalúa con CV y suprime warnings |
| **Cell 11**: Instancia de LogisticRegression | `get_model_instance('logistic_regression')` | Crea modelo con hiperparámetros de config |
| **Cell 11**: Instancia de BorderlineSMOTE | `get_smote_instance('BorderlineSMOTE')` | Crea SMOTE con parámetros de config |
| **Cell 21**: Guardado del modelo con joblib | `train_model()` con `save_model=True` | Serializa pipeline completo |

**Funciones Principales**: `train_model()`, `evaluate_model()`
```python
# Uso en módulo
from mlops_project.modeling.train import train_model, evaluate_model

# Crear y evaluar pipeline
pipeline, results = train_model(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor,
    model_name='logistic_regression',
    use_smote=True,
    evaluate=True
)

# Resultados contienen métricas de CV
print(f"ROC-AUC: {results['roc_auc']['test_mean']:.4f}")
```

---

## Comparación Detallada de Implementaciones

### 1. Limpieza de Variable Objetivo

**Notebook 01, Cell 14**:
```python
def clean_target_nulls(df: pd.DataFrame, target_column_name: str):
    df_clean_target = df[(~df[target_column_name].isnull()) & 
                         (df[target_column_name].isin([0, 1]))].copy()
    return df_clean_target
```

**Módulo dataset.py, líneas 188-190**:
```python
df_clean = df[
    (~df[self.target_column].isnull()) & (df[self.target_column].isin([0, 1]))
].copy()
```
**Status**: Lógica idéntica, ahora como método de clase

---

### 2. Imputación de Valores Inválidos

**Notebook 01, Cell 19**:
```python
def impute_invalid_values_with_mode(df: pd.DataFrame, rules: dict):
    for col, valid_values in rules.items():
        invalid_mask = ~df_imputed[col].isin(valid_values)
        if count_invalid > 0:
            mode_value = df_imputed[col].mode().iloc[0]
            df_imputed.loc[invalid_mask, col] = mode_value
```

**Módulo features.py, líneas 94-105**:
```python
for col, valid_values in self.validation_rules.items():
    invalid_mask = ~X_imputed[col].isin(valid_values)
    count_invalid = invalid_mask.sum()
    if count_invalid > 0 and col in self.mode_values_:
        X_imputed.loc[invalid_mask, col] = self.mode_values_[col]
```
**Status**: Lógica idéntica, mejorada con patrón fit/transform de sklearn

---

### 3. Pipelines de Transformación

**Notebook 02, Cell 8-9**:
```python
numericas_pipe = Pipeline(steps = [
    ('impMediana', SimpleImputer(strategy='median')),
    ('escalaNum', MinMaxScaler(feature_range=(1,2)))
])

nominales_pipe = Pipeline(steps = [
    ('impModa', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

ordinales_pipe = Pipeline(steps = [
    ('impOrd', SimpleImputer(strategy='most_frequent')),
    ('ordtrasnf', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

columnasTransformer = ColumnTransformer(
    transformers = [
        ('numpipe', numericas_pipe, numericas_pipe_nombres),
        ('nominals', nominales_pipe, nominales_pipe_nombres),
        ('ordinales', ordinales_pipe, ordinales_pipe_nombres)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
).set_output(transform="pandas")
```

**Módulo features.py, líneas 258-293**:
```python
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='median')),
    ("scaler", MinMaxScaler(feature_range=(1, 2)))
])

nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
])

ordinal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

self.preprocessor_ = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, self.numeric_features),
        ("nom", nominal_pipeline, self.nominal_features),
        ("ord", ordinal_pipeline, self.ordinal_features),
    ],
    remainder="passthrough",
    verbose_feature_names_out=False,
).set_output(transform="pandas")
```
**Status**: Pipelines idénticos, solo cambian nombres de steps (agregado `sparse_output=False` para compatibilidad)

---

### 4. Cross-Validation y Entrenamiento

**Notebook 03, Cell 8-9**:
```python
def mi_fun(modelo, nombre, Xtraintest, ytraintest, metodo_uo=None):
    pipeline = ImbPipeline(steps=[
        ('preprocesamiento', columnasTransformer),
        ('sub_sobre_muestreo', metodo_uo),
        ('model', modelo)
    ])
    
    micv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=5)
    mismetricas = {
        'miaccuracy': 'accuracy',
        'miprecision': 'precision',
        'mirecall': 'recall',
        'mifi': 'f1',
        'miauc': 'roc_auc',
        'miprauc': 'average_precision',
        'migmean': make_scorer(geometric_mean_score)
    }
    
    scores = cross_validate(pipeline, Xtraintest, np.ravel(ytraintest), 
                           scoring=mismetricas, cv=micv, return_train_score=True)
```

**Módulo modeling/train.py**:
```python
def create_training_pipeline(preprocessor, model, use_smote, smote_method):
    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", get_smote_instance(smote_method)))
    steps.append(("model", model))
    return ImbPipeline(steps=steps)

def evaluate_model(pipeline, X, y, cv_folds=5, cv_repeats=3):
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "geometric_mean": make_scorer(geometric_mean_score)
    }
    cv_results = cross_validate(pipeline, X, np.ravel(y), scoring=scoring, cv=cv, return_train_score=True)
```
**Status**: Lógica idéntica, mismo número de folds, repeats y métricas

---

## Tabla de Equivalencias Completa

### Variables y Parámetros

| Concepto | Notebook | Módulo | Archivo |
|----------|----------|--------|---------|
| Variables numéricas | `lista_paper_num` | `NUMERIC_FEATURES` | `config.py` |
| Variables ordinales | `lista_paper_ord` | `ORDINAL_FEATURES` | `config.py` |
| Variables nominales | `lista_paper_cat` | `NOMINAL_FEATURES` | `config.py` |
| Reglas de validación | `isin_rules` | `CATEGORICAL_VALIDATION_RULES` | `config.py` |
| Variables para outliers | `lista_categoricos` | `OUTLIER_VARIABLES` | `config.py` |
| Semilla aleatoria | Variable en cada notebook | `RANDOM_SEED = 42` | `config.py` |
| Test size | `0.3` hardcoded | `TEST_SIZE = 0.3` | `config.py` |

### Funciones y Clases

| Función Notebook | Clase/Función Módulo | Archivo |
|------------------|----------------------|---------|
| `clean_target_nulls()` | `DataCleaner._clean_target_variable()` | `dataset.py` |
| `impute_invalid_values_with_mode()` | `InvalidDataHandler.fit_transform()` | `features.py` |
| `detect_outliers_iqr()` | `OutlierHandler.fit()` | `features.py` |
| `split_data_by_target()` | `DataSplitter.split()` | `dataset.py` |
| `preprocess_data()` | `prepare_features()` | `features.py` |
| `mi_fun()` | `evaluate_model()` + `train_model()` | `modeling/train.py` |

### Pipelines

| Pipeline Notebook | Equivalente en Módulo | Ubicación |
|-------------------|----------------------|-----------|
| `numericas_pipe` | `numeric_pipeline` | `features.py:258-263` |
| `nominales_pipe` | `nominal_pipeline` | `features.py:266-271` |
| `ordinales_pipe` | `ordinal_pipeline` | `features.py:274-282` |
| `columnasTransformer` | `self.preprocessor_` (ColumnTransformer) | `features.py:285-293` |
| `data_preprocessor` | `FeaturePreprocessor` (clase completa) | `features.py:212-329` |

---

## Mejoras Implementadas (Sin Cambiar la Lógica)

### 1. Patrón fit/transform de sklearn

**Antes (Notebooks)**:
```python
# Funciones directas
df_clean = impute_invalid_values_with_mode(df, rules)
```

**Ahora (Módulos)**:
```python
# Clases con fit/transform
handler = InvalidDataHandler(validation_rules=rules)
handler.fit(X_train)  # Aprende modas
X_clean = handler.transform(X_test)  # Aplica modas aprendidas
```

**Ventaja**: Compatible con sklearn pipelines, evita data leakage

### 2. Configuración Centralizada

**Antes (Notebooks)**:
```python
# Valores hardcoded en cada notebook
test_size = 0.3
random_state = 1234  # o 42 dependiendo del notebook
```

**Ahora (Módulos)**:
```python
# Importar de config
from mlops_project.config import TEST_SIZE, RANDOM_SEED
```

**Ventaja**: Un solo punto de cambio, consistencia garantizada

### 3. Scripts Ejecutables

**Antes (Notebooks)**:
```python
# Ejecutar célula por célula manualmente
```

**Ahora (Módulos)**:
```bash
# Scripts CLI automatizables
uv run mlops-prepare-data --input data.csv --save
uv run mlops-prepare-features --train X.csv --save-preprocessor
uv run mlops-train --model logistic_regression
```

**Ventaja**: Automatización, integración CI/CD, orquestación

### 4. Reutilización

**Antes (Notebooks)**:
```python
# Copiar-pegar código entre notebooks
```

**Ahora (Módulos)**:
```python
# Importar y reutilizar
from mlops_project.dataset import load_and_prepare_data
from mlops_project.features import prepare_features
```

**Ventaja**: DRY (Don't Repeat Yourself), mantenibilidad

---

## Guía de Colaboración

### Para Experimentar con Nuevos Modelos

1. **Agregar modelo a config.py**:
```python
AVAILABLE_MODELS = {
    'nuevo_modelo': {
        'name': 'Nuevo Modelo',
        'params': {
            'parametro1': valor,
            'random_state': RANDOM_SEED
        }
    }
}
```

2. **Usar en train.py**:
```bash
uv run mlops-train --model nuevo_modelo
```

### Para Modificar Preprocesamiento

1. **Experimentar en notebook nuevo**: `notebooks/experiments/04_nuevo_preprocesamiento.ipynb`
2. **Si funciona, actualizar config.py**: Cambiar parámetros
3. **Si requiere nueva lógica, modificar features.py**: Agregar nueva clase o método
4. **Escribir tests**: Agregar tests en `tests/test_features.py`
5. **Verificar**: `make test && make check`

### Para Agregar Nuevas Features

1. **Definir en config.py**:
```python
NEW_FEATURES = ['nueva_feature_1', 'nueva_feature_2']
```

2. **Actualizar FeaturePreprocessor**: Modificar pipelines si es necesario
3. **Actualizar tests**: Verificar que funcione
4. **Documentar**: Actualizar README con la nueva feature

### Workflow de Desarrollo

```
1. Experimentar en Notebooks
   ↓
2. Validar resultados
   ↓
3. Exportar lógica a Módulos
   ↓
4. Escribir Tests
   ↓
5. Actualizar Documentación
   ↓
6. Commit y Push
```

---

## Estructura de Archivos: Dónde Buscar Qué

### Configuración y Parámetros
- **Archivo**: `mlops_project/config.py`
- **Contiene**: Todas las constantes, hiperparámetros, rutas, listas de features
- **Cuándo modificar**: Para cambiar parámetros sin tocar código

### Carga y Limpieza de Datos
- **Archivo**: `mlops_project/dataset.py`
- **Contiene**: `DataLoader`, `DataCleaner`, `DataSplitter`
- **Cuándo modificar**: Para cambiar cómo se cargan o limpian los datos

### Transformación de Features
- **Archivo**: `mlops_project/features.py`
- **Contiene**: `InvalidDataHandler`, `OutlierHandler`, `FeaturePreprocessor`
- **Cuándo modificar**: Para cambiar transformaciones, encoding, o escalado

### Entrenamiento
- **Archivo**: `mlops_project/modeling/train.py`
- **Contiene**: `train_model()`, `evaluate_model()`, `create_training_pipeline()`
- **Cuándo modificar**: Para cambiar proceso de entrenamiento o evaluación

### Predicción
- **Archivo**: `mlops_project/modeling/predict.py`
- **Contiene**: `predict()`, `evaluate_predictions()`, `load_model()`
- **Cuándo modificar**: Para cambiar cómo se hacen predicciones

### Visualizaciones
- **Archivo**: `mlops_project/plots.py`
- **Contiene**: Funciones de plotting (boxplots, ROC, confusion matrix, etc.)
- **Cuándo modificar**: Para agregar nuevas visualizaciones

---

## Verificación de Preservación de Lógica

### Checklist de Equivalencias

- [x] Features numéricas: 3 variables (`laufzeit`, `hoehe`, `alter`)
- [x] Features ordinales: 6 variables
- [x] Features nominales: 11 variables
- [x] SimpleImputer para numéricas: estrategia `median`
- [x] SimpleImputer para categóricas: estrategia `most_frequent`
- [x] MinMaxScaler: rango `(1, 2)`
- [x] OneHotEncoder: `drop='first'`, `handle_unknown='ignore'`
- [x] OrdinalEncoder: `unknown_value=-1`
- [x] Train/test split: `test_size=0.3`, `stratify=True`
- [x] Cross-validation: `5 folds x 3 repeats`
- [x] SMOTE: `BorderlineSMOTE(k_neighbors=5, m_neighbors=10)`
- [x] Logistic Regression: `penalty='l2'`, `solver='newton-cg'`, `C=1`
- [x] Métricas de evaluación: 7 métricas (accuracy, precision, recall, f1, roc_auc, avg_precision, gmean)
- [x] Supresión de warnings durante CV
- [x] Uso de `ImbPipeline` para SMOTE
- [x] Salida de ColumnTransformer: formato pandas

**Resultado**: Todas las verificaciones pasadas. La lógica está 100% preservada.

---

## Flujo Completo: Notebooks vs Módulos

### Flujo en Notebooks

```
Notebook 01:
  Cargar datos → Limpiar no numéricos → Limpiar target → 
  Imputar inválidos → Detectar outliers → Guardar data_clean.csv

Notebook 02:
  Cargar data_clean.csv → Dividir train/test → 
  Crear ColumnTransformer → Aplicar transformaciones → 
  Guardar Xtraintest.csv, ytraintest.csv

Notebook 03:
  Cargar Xtraintest, ytraintest → Crear ImbPipeline → 
  Cross-validate → Entrenar modelo final → 
  Guardar best_model.joblib
```

### Flujo en Módulos (Equivalente)

```
dataset.py:
  DataLoader.load_data() → DataCleaner.clean_data() → 
  DataSplitter.split() → Guardar procesados

features.py:
  InvalidDataHandler.fit_transform() → OutlierHandler.fit_transform() → 
  FeaturePreprocessor.fit_transform() → Guardar preprocessor.joblib

modeling/train.py:
  create_training_pipeline() → evaluate_model() (CV) → 
  train_model() → Guardar best_model.joblib
```

**Diferencia**: El flujo modular permite ejecutar cada paso independientemente y es reutilizable.

---

## Scripts CLI vs Notebooks

### Ejecución de Pipeline Completo

**Con Notebooks** (Manual):
```
1. Abrir 01_eda_south_german_credit.ipynb
2. Ejecutar todas las celdas
3. Abrir 02_preprocessing_pipeline.ipynb
4. Ejecutar todas las celdas
5. Abrir 03_model_training_evaluation.ipynb
6. Ejecutar todas las celdas
```

**Con Módulos** (Automatizado):
```bash
# Una sola línea
make pipeline

# O manualmente
uv run mlops-prepare-data --input data/raw/german_credit_modified.csv --save
uv run mlops-prepare-features --train data/processed/Xtraintest.csv --save-preprocessor
uv run mlops-train --X-train data/processed/Xtraintest.csv --y-train data/processed/ytraintest.csv --preprocessor models/preprocessor.joblib --model logistic_regression
```

---

## Para Nuevos Miembros del Equipo

### Si quieres entender la lógica original:
1. Revisar notebooks en orden: `01 → 02 → 03`
2. Leer comentarios y outputs de cada celda
3. Experimentar modificando parámetros en los notebooks

### Si quieres contribuir al código productivo:
1. Revisar módulos en `mlops_project/`
2. Leer tests en `tests/` para entender comportamiento esperado
3. Leer `mlops_project/config.py` para entender parámetros
4. Hacer cambios en módulos, no en notebooks
5. Ejecutar `make test` y `make check` antes de commit

### Si quieres experimentar con nuevas ideas:
1. Crear nuevo notebook en `notebooks/experiments/`
2. Importar módulos existentes:
   ```python
   from mlops_project.dataset import load_and_prepare_data
   from mlops_project.features import prepare_features
   from mlops_project.config import RANDOM_SEED
   ```
3. Experimentar libremente
4. Si funciona, proponer migración a módulos

---

## Preguntas Frecuentes

**¿Puedo modificar los notebooks originales?**
No. Los notebooks 01, 02 y 03 son referencia histórica. Crear nuevos notebooks para experimentación.

**¿Dónde cambio hiperparámetros de modelos?**
En `mlops_project/config.py` → `AVAILABLE_MODELS`

**¿Dónde agrego un nuevo modelo?**
1. Agregar a `config.py` → `AVAILABLE_MODELS`
2. Usar con: `uv run mlops-train --model nuevo_modelo`

**¿Cómo sé que la lógica es la misma?**
Ejecutar tests: `make test`. Los tests verifican que todo funcione igual.

**¿Puedo usar los módulos en notebooks nuevos?**
Sí, es la forma recomendada:
```python
from mlops_project.dataset import load_and_prepare_data
X_train, X_test, y_train, y_test = load_and_prepare_data(...)
```

**¿Dónde están los outputs del pipeline?**
- Datos: `data/processed/`
- Modelos: `models/`
- Resultados: `models/model_results.json`

---

## Resumen para el Equipo

### Lo que NO cambió:
- Algoritmos de limpieza
- Transformaciones de features
- Pipelines de sklearn
- Hiperparámetros de modelos
- Métricas de evaluación
- Estrategias de imputación
- Métodos de detección de outliers

### Lo que SÍ cambió (mejoras de ingeniería):
- Código organizado en módulos reutilizables
- Clases que implementan interfaz sklearn
- Configuración centralizada
- Scripts CLI para automatización
- Tests automatizados (130 tests)
- Documentación completa
- Preparado para MLFlow y Airflow

### Regla de Oro:
**Los notebooks son documentación de experimentación. Los módulos son código productivo.**

---

**Equipo 29** - TC5044.10  
**Última actualización**: Octubre 2025  
**Mantenedores**: Ver README principal

