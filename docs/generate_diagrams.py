#!/usr/bin/env python3
"""
Script para generar diagramas UML de clases desde código Python.

Este script:
1. Analiza el código Python del proyecto
2. Genera diagramas UML de clases automáticamente
3. Exporta las imágenes localmente usando graphviz

Usa pyreverse (pylint) para extraer clases y graphviz para renderizar.
"""

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def find_graphviz_bin() -> Optional[Path]:
    """Busca el ejecutable dot de graphviz."""
    try:
        result = subprocess.run(
            ["which", "dot"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except Exception:
        pass
    
    # Rutas comunes
    common_paths = [
        Path("/usr/local/bin/dot"),
        Path("/usr/bin/dot"),
        Path("/opt/homebrew/bin/dot"),  # macOS con Homebrew
    ]
    
    for path in common_paths:
        if path.exists() and path.is_file():
            return path
    
    return None


def check_graphviz_installed() -> Tuple[bool, str]:
    """Verifica si graphviz está instalado."""
    dot_path = find_graphviz_bin()
    if dot_path:
        return True, str(dot_path)
    return False, "Graphviz no encontrado. Instala con: brew install graphviz (macOS) o apt-get install graphviz (Linux)"


def generate_class_diagram(
    module_path: str,
    output_name: str,
    output_dir: Path,
    all_classes: bool = True,
    show_attributes: bool = True,
    show_methods: bool = True,
) -> Tuple[bool, str]:
    """
    Genera diagrama de clases para un módulo usando pyreverse.
    
    Args:
        module_path: Ruta al módulo o paquete (ej: "mlops_project.dataset")
        output_name: Nombre del archivo de salida (sin extensión)
        output_dir: Directorio de salida
        all_classes: Incluir todas las clases (incluyendo internas de librerías)
        show_attributes: Mostrar atributos
        show_methods: Mostrar métodos
    
    Returns:
        Tuple (success, message)
    """
    dot_path = find_graphviz_bin()
    if not dot_path:
        installed, msg = check_graphviz_installed()
        return False, msg
    
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).parent.parent
    
    # Construir comando pyreverse
    # pyreverse genera diagramas de clases y paquetes
    cmd = [
        "uv", "run", "pyreverse",
        "-o", "dot",  # Formato de salida: dot
        "-p", output_name,  # Prefijo para archivos generados
        "-ASmy",  # Opciones: A (all), S (show relations), m (modules), y (generate)
    ]
    
    if not all_classes:
        # Filtrar solo clases principales (sin clases internas de librerías)
        cmd.append("--filter-mode=INCLUDE")  # Pero esto puede no funcionar bien
    
    cmd.append(module_path)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # pyreverse genera archivos en el directorio actual
        # Buscar el archivo generado
        dot_file = project_root / f"classes_{output_name}.dot"
        if not dot_file.exists():
            # Intentar otros nombres posibles
            for possible_name in [
                f"classes_{output_name}.dot",
                f"{output_name}_classes.dot",
                "classes.dot",
            ]:
                possible_file = project_root / possible_name
                if possible_file.exists():
                    dot_file = possible_file
                    break
        
        if not dot_file.exists():
            error_msg = result.stderr or result.stdout
            return False, f"No se generó archivo .dot. Error: {error_msg}"
        
        # Convertir .dot a PNG usando graphviz
        png_output = output_dir / f"{output_name}.png"
        
        dot_result = subprocess.run(
            [str(dot_path), "-Tpng", "-o", str(png_output), str(dot_file)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        
        # Limpiar archivos .dot temporales (tanto classes_ como packages_)
        for temp_file in project_root.glob("*.dot"):
            try:
                if temp_file.name.startswith(("classes_", "packages_")) or "diagrama" in temp_file.name.lower():
                    temp_file.unlink()
            except Exception:
                pass
        
        if dot_result.returncode == 0 and png_output.exists():
            # Verificar que la imagen es válida
            from PIL import Image
            try:
                img = Image.open(png_output)
                img.verify()
                img = Image.open(png_output)  # Reabrir después de verify
                return True, f"Diagrama generado: {png_output} ({img.size[0]}x{img.size[1]})"
            except Exception as e:
                return False, f"Imagen generada pero inválida: {e}"
        else:
            error = dot_result.stderr or dot_result.stdout
            return False, f"Error generando PNG: {error}"
            
    except subprocess.TimeoutExpired:
        return False, "Timeout generando diagrama"
    except FileNotFoundError:
        return False, "pyreverse no encontrado. Asegúrate de tener pylint instalado"
    except Exception as e:
        return False, f"Error ejecutando pyreverse: {e}"


def filter_diagram_classes(dot_file: Path, project_classes_only: bool = True) -> Path:
    """
    Filtra el diagrama para mostrar solo clases del proyecto.
    
    Args:
        dot_file: Archivo .dot generado
        project_classes_only: Si True, solo mantener clases de mlops_project
    
    Returns:
        Path al archivo .dot filtrado
    """
    if not project_classes_only:
        return dot_file
    
    content = dot_file.read_text(encoding="utf-8")
    
    # Filtrar líneas que contengan clases externas comunes
    filtered_lines = []
    skip_next = False
    
    for line in content.split("\n"):
        # Mantener estructura básica del grafo
        if line.strip().startswith(("digraph", "graph", "}", "{", "node", "edge")):
            filtered_lines.append(line)
            continue
        
        # Mantener clases que empiecen con mlops_project
        if "mlops_project" in line or "label=" in line:
            filtered_lines.append(line)
            continue
        
        # Omitir clases externas conocidas (puedes ajustar esta lógica)
        external_patterns = [
            "sklearn",
            "pandas",
            "numpy",
            "pathlib",
            "typing",
        ]
        
        if any(pattern in line.lower() for pattern in external_patterns):
            continue
        
        filtered_lines.append(line)
    
    filtered_file = dot_file.parent / f"{dot_file.stem}_filtered.dot"
    filtered_file.write_text("\n".join(filtered_lines), encoding="utf-8")
    return filtered_file


def main():
    """Función principal."""
    project_root = Path(__file__).parent.parent
    diagrams_dir = project_root / "docs" / "diagrams"
    images_dir = diagrams_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar graphviz
    graphviz_installed, graphviz_msg = check_graphviz_installed()
    if not graphviz_installed:
        print("ERROR:", graphviz_msg)
        print("\nInstala graphviz:")
        print("  macOS: brew install graphviz")
        print("  Linux: sudo apt-get install graphviz")
        print("  Windows: choco install graphviz")
        sys.exit(1)
    
    print("=" * 60)
    print("GENERACIÓN DE DIAGRAMAS UML DESDE CÓDIGO PYTHON")
    print("=" * 60)
    print(f"Graphviz encontrado: {graphviz_msg}\n")
    
    # Módulos a diagramar
    modules = [
        ("mlops_project.dataset", "diagrama_dataset", "Clases de dataset.py"),
        ("mlops_project.features", "diagrama_features", "Clases de features.py"),
        ("mlops_project", "diagrama_completo", "Vista completa del proyecto"),
    ]
    
    results = []
    
    for module_path, output_name, description in modules:
        print(f"\n[{description}]")
        print(f"Generando diagrama para {module_path}...", end=" ")
        
        success, msg = generate_class_diagram(
            module_path=module_path,
            output_name=output_name,
            output_dir=images_dir,
            all_classes=False,  # Solo clases del módulo
            show_attributes=True,
            show_methods=True,
        )
        
        if success:
            print(f"✓ {msg}")
            results.append((output_name, True, msg))
        else:
            print(f"✗ {msg}")
            results.append((output_name, False, msg))
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    for output_name, success, msg in results:
        status = "✓" if success else "✗"
        print(f"{status} {output_name}: {msg if success else 'Error generando'}")
    
    print("\n" + "=" * 60)
    
    if success_count == total_count:
        print("✓ Todos los diagramas generados exitosamente")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"⚠ {success_count}/{total_count} diagramas generados exitosamente")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
