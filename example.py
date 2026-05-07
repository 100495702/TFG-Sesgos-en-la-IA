"""
Script de ejemplo para demostrar el uso del generador de evaluaciones
Ejecutar este archivo para obtener una prueba rápida
"""

import sys
import json
from pathlib import Path

# Agregar al path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_evaluations import GeneradorEvaluaciones


def ejemplo_basico():
    """Ejecutar ejemplo básico de generación"""
    print("="*60)
    print("EJEMPLO: Generación de Evaluaciones")
    print("="*60)
    
    # Ruta del dataset
    dataset_path = Path("/mnt/users_backup/alejandrol_tfg/TFG/data/dataset.json")
    
    if not dataset_path.exists():
        print(f"Error: {dataset_path} no encontrado")
        return
    
    print(f"\n1. Cargando dataset desde: {dataset_path}")
    
    # Crear generador
    generador = GeneradorEvaluaciones(str(dataset_path))
    print(f"   ✓ Dataset cargado: {len(generador.dataset)} prompts")
    print(f"   ✓ Modelos disponibles: {list(generador.config_modelos.keys())}")
    
    # Generar evaluaciones para 1 prompt y 1 modelo como ejemplo
    print(f"\n2. Generando evaluaciones (ejemplo: 1 prompt, 1 modelo)...")
    resultados = generador.generar_evaluaciones(
        modelos=["Llama_3_8B_Instruct"],
        limite_prompts=1
    )
    
    if resultados:
        print(f"   ✓ Generadas {len(resultados)} evaluaciones")
        
        # Mostrar primer resultado
        resultado = resultados[0]
        print(f"\n3. Ejemplo de resultado:")
        print(f"   ID Prueba: {resultado['id_prueba']}")
        print(f"   Dominio: {resultado['dominio']}")
        print(f"   Modelo: {resultado['modelo_evaluado']}")
        print(f"   Plantilla: {resultado['plantilla_prompt'][:60]}...")
        
        # Mostrar métricas base
        control = resultado['evaluaciones']['control_base']
        print(f"\n   Control Base:")
        print(f"     - Longitud: {control['metricas_absolutas']['longitud_palabras']} palabras")
        print(f"     - Polaridad: {control['metricas_absolutas']['score_polaridad_roberta']}")
        print(f"     - Paternalismo: {control['metricas_absolutas']['score_paternalismo_juez']}")
        
        # Mostrar primera variante
        variantes = resultado['evaluaciones']['variantes_contrafactuales']
        if variantes:
            primera_var = list(variantes.items())[0]
            nombre_var, datos_var = primera_var
            print(f"\n   {nombre_var}:")
            print(f"     - Condición: {datos_var['variable_insertada']}")
            print(f"     - Longitud: {datos_var['metricas_absolutas']['longitud_palabras']} palabras")
            print(f"     - Polaridad: {datos_var['metricas_absolutas']['score_polaridad_roberta']}")
            print(f"     - Paternalismo: {datos_var['metricas_absolutas']['score_paternalismo_juez']}")
            
            if 'metricas_delta' in datos_var:
                deltas = datos_var['metricas_delta']
                print(f"     - Delta Longitud: {deltas['delta_longitud']}")
                print(f"     - Delta Polaridad: {deltas['delta_polaridad']}")
                print(f"     - Delta Paternalismo: {deltas['delta_paternalismo']}")
    
    # Guardar resultados
    print(f"\n4. Guardando resultados...")
    generador.guardar_resultados()
    
    # Mostrar reporte
    generador.generar_reporte()
    
    print("\n" + "="*60)
    print("EJEMPLO COMPLETADO")
    print("="*60)
    print(f"\nResultados guardados en: {generador.output_path}")
    print("\nPara generar todas las evaluaciones, edita generate_evaluations.py:")
    print("  - Cambia limite_prompts=None")
    print("  - Agregar otros modelos en modelos_a_usar")


def mostrar_estructura():
    """Mostrar estructura del proyecto"""
    print("\n" + "="*60)
    print("ESTRUCTURA DEL PROYECTO")
    print("="*60)
    
    dirs_y_archivos = {
        'data': ['dataset.json', 'prompts.txt'],
        'scripts': ['generate_evaluations.py', 'integracion_modelos.py', 'example.py'],
        'outputs': []  # Se crea automáticamente
    }
    
    for directorio, archivos in dirs_y_archivos.items():
        ruta = PROJECT_ROOT / directorio
        existe = ruta.exists()
        icono = "✓" if existe else "✗"
        print(f"\n{icono} {directorio}/")
        
        if archivos:
            for archivo in archivos:
                archivo_ruta = ruta / archivo
                existe_archivo = archivo_ruta.exists() if existe else False
                icono_archivo = "✓" if existe_archivo else "○"
                print(f"    {icono_archivo} {archivo}")


def mostrar_ayuda():
    """Mostrar mensaje de ayuda"""
    print("\n" + "="*60)
    print("AYUDA - PRÓXIMOS PASOS")
    print("="*60)
    print("""
1. INSTALACIÓN DE DEPENDENCIAS:
   pip install -r requirements.txt

2. CONFIGURACIÓN DE APIs (opcional):
   - Gemini: Set GOOGLE_API_KEY=tu_clave
   - Claude: Set ANTHROPIC_API_KEY=tu_clave

3. GENERAR EVALUACIONES COMPLETAS:
   - Edita: scripts/generate_evaluations.py
   - Línea 360: limite_prompts = None  (cambiar de 2 a None)
   - Línea 357: agregar todos los modelos necesarios
   - Ejecuta: python scripts/generate_evaluations.py

4. ANALIZAR RESULTADOS:
   - Ver: outputs/salidas.json
   - Crear scripts de análisis personalizados

5. INTEGRAR MODELOS REALES:
   - Ver: scripts/integracion_modelos.py
   - Implementar consultas a modelos en generate_evaluations.py
""")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BIENVENIDO A LA EVALUACIÓN DE SESGOS EN LLMs")
    print("="*60)
    
    # Mostrar estructura
    mostrar_estructura()
    
    # Ejecutar ejemplo
    print("\n")
    input("Presiona Enter para ejecutar ejemplo básico...")
    ejemplo_basico()
    
    # Mostrar ayuda
    mostrar_ayuda()
