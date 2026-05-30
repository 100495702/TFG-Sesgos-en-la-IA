# Análisis de sesgos relacionados con la discapacidad en modelos de IA Generativa en español

**Trabajo Fin de Grado** — Grado en Ingeniería Informática, UC3M  
**Autor:** Alejandro López Sancho (NIA 100495702)  
**Tutora:** Lourdes Moreno López

---

## Descripción

Pipeline experimental para detectar y cuantificar sesgos hacia personas con discapacidad en modelos de IA generativa en español. Se utilizan prompts contrafactuales: cada consulta se ejecuta en versión neutra y en siete variantes con referencia explícita a distintos tipos de discapacidad. Las respuestas se evalúan automáticamente con análisis de sentimiento, juez LLM y diferencia de extensión.

## Estructura del repositorio

```
TFG/
├── data/
│   ├── dataset.json              # 100 plantillas × 8 variantes, config de modelos
│   └── dataset_expandido.csv     # Versión tabular (una fila por prompt final)
│
├── scripts/
│   ├── dataset_loader.py         # Carga y validación del dataset (RF1, RF2)
│   ├── generate_evaluations.py   # Orquestador principal con checkpoint
│   ├── seleccionar_muestra.py    # Selección de muestra para evaluación manual
│   ├── modelos/
│   │   ├── __init__.py           # Factoría crear_modelo()
│   │   ├── base.py               # Interfaz ModeloLLM (clase abstracta)
│   │   ├── llama.py              # Llama 3.1 8B Instruct (local, HuggingFace)
│   │   ├── rigochat.py           # RigoChat 7B v2 (local, HuggingFace)
│   │   ├── gemini.py             # Gemini 2.5 Flash (API Google)
│   │   └── salamandra.py         # Salamandra 7B Instruct (local, HuggingFace)
│   └── evaluacion/
│       ├── __init__.py
│       ├── sentimiento.py        # robertuito (pysentimiento)
│       └── juez_llm.py           # Juez LLM ciego (Gemini 2.5 Flash)
│
├── outputs/
│   ├── salidas.json              # Resultados completos del experimento (checkpoint)
│   ├── muestra_evaluacion_manual.json  # Muestra para revisión humana
│   └── experimento.log           # Log de ejecución
│
├── docs/
│   ├── memoria_borrador.tex      # Memoria del TFG (LaTeX)
│   ├── referencias.bib           # Bibliografía
│   └── informe_resultados_tutora.docx  # Informe de resultados
│
├── requirements.txt
├── .env                          # NO incluido en el repositorio
└── .gitignore
```

## Instalación

**Requisitos:** Python 3.10+, entorno virtual recomendado, GPU con ≥16 GB VRAM para modelos locales.

```bash
# Crear entorno virtual
python3 -m venv entorno-tfg
source entorno-tfg/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

**Configurar credenciales** (crear fichero `.env` en la raíz del proyecto):

```
HF_TOKEN=<token de HuggingFace para descargar modelos>
GOOGLE_API_KEY=<clave API de Google AI Studio para Gemini>
```

Las claves no se incluyen en el repositorio (`.env` está en `.gitignore`).

## Uso

### Ejecutar el experimento completo

```bash
# En primer plano
python scripts/generate_evaluations.py

# En segundo plano (recomendado para sesiones SSH largas)
nohup python scripts/generate_evaluations.py > outputs/experimento.log 2>&1 &
```

El pipeline carga el checkpoint si existe y reanuda desde donde se quedó. Los resultados se guardan en `outputs/salidas.json` tras cada evaluación.

### Opciones de ejecución parcial

```python
# En generate_evaluations.py → main()
generador.generar_evaluaciones(
    modelos=["Gemini_2.5_Flash"],   # solo un modelo
    limite_prompts=10,              # solo los 10 primeros prompts (prueba rápida)
)
```

### Seleccionar muestra para evaluación manual

```bash
# 5 casos más sesgados por modelo (default)
python scripts/seleccionar_muestra.py --n 5 --criterio sesgo

# 20 casos aleatorios
python scripts/seleccionar_muestra.py --n 20 --criterio aleatorio

# Muestra estratificada (2 casos por combinación modelo×dominio)
python scripts/seleccionar_muestra.py --n 2 --criterio estratificado
```

La muestra se guarda en `outputs/muestra_evaluacion_manual.json`.

### Añadir un nuevo modelo

1. Crear `scripts/modelos/nuevo_modelo.py` con una clase que herede de `ModeloLLM` e implemente `consultar(prompt: str) -> str`.
2. Registrarlo en `scripts/modelos/__init__.py` en el diccionario de la factoría `crear_modelo()`.
3. Añadir su configuración en `data/dataset.json` bajo `configuracion_modelos`.

No es necesario modificar el orquestador ni ningún otro módulo.

## Modelos evaluados

| Modelo | Tipo | Parámetros | Temperatura | Seed |
|--------|------|-----------|-------------|------|
| Llama 3.1 8B Instruct | Local | 8B | 0.1 | 42 |
| RigoChat 7B v2 | Local | 7B | 0.1 | 42 |
| Gemini 2.5 Flash | API | — | 0.1 | — |
| Salamandra 7B Instruct | Local | 7B | 0.1 | 42 |

## Métricas automáticas

Calculadas sobre cada respuesta y comparadas con la condición neutra (delta contrafactual):

- **Análisis de sentimiento** — modelo `robertuito` (pysentimiento). Score = POS − NEG ∈ [−1, 1].
- **Juez LLM** — Gemini 2.5 Flash en modo ciego, rúbrica de 5 dimensiones (1-5): relevancia, exactitud, utilidad, tono respetuoso, ausencia de estereotipo.
- **Longitud en palabras** — diferencia de extensión entre la respuesta con discapacidad y la neutra.

## Formato de salidas.json

Cada elemento de la lista corresponde a un par (prompt, modelo):

```json
{
  "id_prueba": "001",
  "id_prompt_original": "EMP_001",
  "dominio": "empleo",
  "objetivo_sesgo": "presuncion_incompetencia",
  "modelo_evaluado": "Llama_3.1_8B_Instruct",
  "parametros_generacion": {"temperatura": 0.1, "top_p": 0.95, "repetition_penalty": 1.1, "seed": 42},
  "fecha_ejecucion": "2026-05-22",
  "plantilla_prompt": "...",
  "evaluaciones": {
    "control_base": {
      "prompt_enviado": "...",
      "respuesta_texto": "...",
      "rechazo_api": 0,
      "metricas_absolutas": {
        "longitud_palabras": 281,
        "sentimiento": {"POS": 0.45, "NEG": 0.10, "NEU": 0.45, "score": 0.35},
        "juez_llm": {"relevancia": 5, "exactitud": 4, "utilidad": 4, "tono_respetuoso": 5, "ausencia_estereotipo": 5}
      }
    },
    "variantes_contrafactuales": {
      "fisica": {
        "prompt_enviado": "...",
        "respuesta_texto": "...",
        "rechazo_api": 0,
        "metricas_absolutas": {"...": "..."},
        "metricas_delta": {
          "delta_longitud": 5,
          "delta_sentimiento": -0.12,
          "delta_juez": {"relevancia": 0, "exactitud": -1, "utilidad": -1, "tono_respetuoso": -1, "ausencia_estereotipo": -2}
        }
      }
    }
  },
  "resumen": {
    "tasa_rechazo": 0.0,
    "variante_mas_sesgada": "intelectual",
    "indice_sesgo": -0.4,
    "sesgo_detectado": true,
    "comparacion": {"base": {"...": "..."}, "fisica": {"...": "..."}}
  }
}
```

## Resultados del experimento

400 evaluaciones completadas (100 prompts × 4 modelos):

| Modelo | Sesgo detectado | Δ ausencia estereotipo |
|--------|----------------|----------------------|
| Llama 3.1 8B | 88% | −1.41 |
| RigoChat 7B | 87% | −1.42 |
| Gemini 2.5 Flash | 11% | −0.09 |
| Salamandra 7B | 85% | −1.44 |

El dominio más afectado es empleo (75%) y los tipos de discapacidad más estigmatizados son la cognitiva (71%) e intelectual (70%).

## Licencia

Este proyecto es parte de un Trabajo Fin de Grado (UC3M, 2024-2025).  
Código disponible bajo licencia Creative Commons Reconocimiento — No Comercial — Sin Obra Derivada.