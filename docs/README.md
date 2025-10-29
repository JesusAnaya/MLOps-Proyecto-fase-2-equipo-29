# Documentación del Proyecto MLOps

Este directorio contiene documentación adicional del proyecto, incluyendo diagramas UML generados automáticamente desde el código Python.

## Estructura

```
docs/
├── README.md              # Este archivo
├── diagrams/              # Diagramas UML de clases
│   └── images/            # Imágenes generadas (PNG)
│       ├── diagrama_dataset.png    # Clases del módulo dataset.py
│       ├── diagrama_features.png   # Clases del módulo features.py
│       └── diagrama_completo.png   # Vista general de todo el proyecto
└── generate_diagrams.py  # Script para generar diagramas desde código
```

## Diagramas UML

Los diagramas UML se generan **automáticamente** desde el código Python del proyecto usando `pyreverse` (parte de pylint). Esto garantiza que los diagramas siempre reflejen la estructura actual del código.

### Diagramas Disponibles

1. **`diagrama_dataset.png`**
   - Clases del módulo `mlops_project.dataset`
   - Clases: `DataLoader`, `DataCleaner`, `DataSplitter`
   - Muestra relaciones y herencias

2. **`diagrama_features.png`**
   - Clases del módulo `mlops_project.features`
   - Clases: `InvalidDataHandler`, `OutlierHandler`, `FeaturePreprocessor`
   - Muestra herencia de `BaseEstimator` y `TransformerMixin` (sklearn)
   - Indica compatibilidad con sklearn pipelines

3. **`diagrama_completo.png`**
   - Vista general de toda la arquitectura del proyecto
   - Todas las clases y sus relaciones
   - Dependencias entre módulos

### Características de los Diagramas

- **Generación automática**: Los diagramas se generan desde el código, no se editan manualmente
- **Siempre actualizados**: Ejecutar el script regenera los diagramas con la estructura actual
- **Local**: No requiere conexión a internet ni servicios externos
- **Relaciones reales**: Muestra las relaciones reales entre clases según el código

## Generación de Diagramas

### Requisitos

1. **Python y UV** (ya instalado en el proyecto)
2. **pylint** (incluido en dependencias del proyecto)
3. **graphviz** (requiere instalación del sistema)

#### Instalar Graphviz

**macOS:**
```bash
brew install graphviz
```

**Linux:**
```bash
sudo apt-get install graphviz
```

**Windows:**
```powershell
choco install graphviz
```

### Uso del Script de Generación

El script `generate_diagrams.py` analiza el código Python y genera diagramas UML automáticamente:

```bash
# Ejecutar desde la raíz del proyecto
cd /ruta/al/proyecto
uv run python docs/generate_diagrams.py

# O con make
make validate-diagrams
```

**El script realiza**:
1. Verifica que graphviz esté instalado
2. Analiza el código Python usando `pyreverse`
3. Genera archivos `.dot` intermedios
4. Convierte a imágenes PNG usando `graphviz`
5. Valida que las imágenes sean correctas
6. Guarda en `docs/diagrams/images/`

### Ejemplo de Salida

```
============================================================
GENERACIÓN DE DIAGRAMAS UML DESDE CÓDIGO PYTHON
============================================================
Graphviz encontrado: /usr/local/bin/dot

[Clases de dataset.py]
Generando diagrama para mlops_project.dataset... [OK] Diagrama generado: .../diagrama_dataset.png (2314x1087)

[Clases de features.py]
Generando diagrama para mlops_project.features... [OK] Diagrama generado: .../diagrama_features.png (3157x2122)

[Vista completa del proyecto]
Generando diagrama para mlops_project... [OK] Diagrama generado: .../diagrama_completo.png (4690x2363)

============================================================
RESUMEN
============================================================
[OK] diagrama_dataset: Diagrama generado exitosamente
[OK] diagrama_features: Diagrama generado exitosamente
[OK] diagrama_completo: Diagrama generado exitosamente

============================================================
[OK] Todos los diagramas generados exitosamente
============================================================
```

## Actualización de Diagramas

### Cuándo Regenerar

Regenera los diagramas cuando:
- Agregues nuevas clases al proyecto
- Modifiques la estructura de clases existentes
- Cambies relaciones entre clases (herencia, composición)
- Refactorices módulos

### Cómo Regenerar

Simplemente ejecuta el script:

```bash
uv run python docs/generate_diagrams.py
```

Los diagramas se regenerarán automáticamente desde el código actual. **No necesitas editar archivos manualmente** - todo se genera desde el código Python.

### Integración en Workflow

**Pre-commit** (recomendado):
```bash
# En .git/hooks/pre-commit
#!/bin/bash
uv run python docs/generate_diagrams.py
git add docs/diagrams/images/*.png
```

**CI/CD** (futuro):
El script puede integrarse en pipelines para asegurar que los diagramas estén siempre actualizados.

## Ventajas de Generación Automática

### vs Diagramas Manuales

- **Siempre sincronizados** con el código
- **No se olvidan actualizar** (se regeneran automáticamente)
- **Reflejan relaciones reales** del código
- **Menos trabajo manual** para mantener

### vs Servicios Online

- **Funciona offline** - no requiere internet
- **Más rápido** - generación local inmediata
- **Más confiable** - no depende de servicios externos
- **Privado** - código no se envía fuera

## Herramientas Usadas

### pyreverse (pylint)

- **Propósito**: Analiza código Python y extrae estructura de clases
- **Genera**: Archivos `.dot` con información de clases y relaciones
- **Incluido en**: `pylint` (ya está en dependencias del proyecto)

### Graphviz

- **Propósito**: Convierte archivos `.dot` a imágenes PNG/SVG
- **Instalación**: Requiere instalación del sistema (no Python package)
- **Comando**: `dot -Tpng input.dot -o output.png`

## Solución de Problemas

### Error: "Graphviz no encontrado"

```bash
# macOS
brew install graphviz

# Verificar instalación
which dot
# Debe mostrar: /usr/local/bin/dot (o similar)
```

### Error: "pyreverse no encontrado"

```bash
# Instalar pylint (debería estar ya instalado)
uv add pylint
```

### Diagramas muy grandes o pequeños

Los diagramas se generan automáticamente según el tamaño del código. Si son demasiado grandes:
- Considera generar diagramas por módulo (ya implementado)
- Usa zoom en tu visor de imágenes

### Diagramas muestran demasiadas clases externas

`pyreverse` puede incluir clases de librerías externas. El script ya filtra algunas, pero puedes ajustar los parámetros en `generate_diagrams.py` si necesitas más control.

## Recursos Adicionales

- [Documentación pylint/pyreverse](https://pylint.pycqa.org/)
- [Documentación Graphviz](https://graphviz.org/)
- [Formato DOT (Graphviz)](https://graphviz.org/doc/info/lang.html)

## Mantenimiento

- **Responsable**: Todo el equipo
- **Frecuencia**: Regenerar antes de commits importantes o cuando cambies estructura de clases
- **Automatización**: Considerar pre-commit hook para regenerar automáticamente

---

**Nota**: Los diagramas son generados automáticamente desde el código. Mantenerlos actualizados es tan simple como ejecutar el script cuando cambies la estructura de clases.
