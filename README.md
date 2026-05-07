# TFG - Evaluación de Sesgos en Modelos LLM para Personas con Discapacidad

## Estructura del Proyecto

```
TFG/
├── data/                      # Archivos de entrada
│   ├── dataset.json          # Dataset con 100 prompts
│   ├── prompts.txt           # Prompts originales (referencia)
│   └── new_dataset.json      # Backup de dataset anterior
│
├── scripts/                   # Scripts de procesamiento
│   └── generate_evaluations.py    # Script principal de generación de evaluaciones
│
├── outputs/                   # Archivos de salida
│   └── salidas.json          # Evaluaciones generadas por los modelos
│
└── README.md                  # Este archivo
```

## Dataset

El archivo `data/dataset.json` contiene:

- **100 prompts** distribuidos en 5 dominios:
  - Empleo (001-020): 20 prompts
  - Educación (021-040): 20 prompts
  - Salud (041-060): 20 prompts
  - Administrativo (061-080): 20 prompts
  - Ocio (081-100): 20 prompts

- **8 variaciones** de discapacidad para cada prompt:
  - Base (sin variación)
  - Física (en silla de ruedas)
  - Visual Parcial (con visión reducida)
  - Visual Completa (con ceguera total)
  - Auditiva Parcial (con audición reducida)
  - Auditiva Completa (con sordera total)
  - Intelectual (con síndrome de Down)
  - Cognitiva (con discapacidad cognitiva)

## Configuración de Modelos

El dataset incluye configuración para 4 modelos LLM:

1. **Llama_3_8B_Instruct** - Ejecución local (HuggingFace)
   - Temperatura: 0.1
   - Top-p: 0.95
   - Repetition penalty: 1.1

2. **RigoChat_7B_v2** - Ejecución local (HuggingFace)
   - Temperatura: 0.1
   - Top-p: 0.95
   - Repetition penalty: 1.15

3. **Gemini_1.5_Flash** - API Google
   - Temperatura: 0.1
   - Top-p: 0.95

4. **Claude_3.5_Sonnet** - API Anthropic
   - Temperatura: 0.1
   - Top-p: 0.95

## Instalación

### Requisitos

- Python 3.8+
- pip

### Dependencias

```bash
# Instalar dependencias básicas
pip install -r requirements.txt

# Para soporte de modelos locales (transformers)
pip install torch transformers

# Para APIs (opcional)
pip install google-generativeai anthropic
```

## Uso

### Generar Evaluaciones

```bash
cd scripts
python generate_evaluations.py
```

El script:
1. Carga el dataset desde `data/dataset.json`
2. Para cada prompt y variación:
   - Genera el prompt completo con la condición
   - Obtiene respuesta del modelo
   - Calcula métricas (longitud, polaridad, paternalismo)
   - Calcula deltas respecto a línea base
3. Almacena resultados en `outputs/salidas.json`

### Configuración del Script

En `generate_evaluations.py`, edita estas líneas para personalizar:

```python
# Cambiar modelos a evaluar
modelos_a_usar = ["Llama_3_8B_Instruct", "Gemini_1.5_Flash"]

# Cambiar número de prompts a procesar (None = todos)
limite_prompts = None  # O un número como 10 para testing
```

## Formato de Salida

El archivo `outputs/salidas.json` contiene:

```json
{
  "id_prueba": "001",
  "id_prompt_original": "001",
  "dominio": "Empleo",
  "modelo_evaluado": "Llama_3_8B_Instruct",
  "configuracion": {
    "temperatura": 0.1,
    "top_p": 0.95,
    "repetition_penalty": 1.1,
    "seed": 42
  },
  "plantilla_prompt": "Redacta un correo rechazando tras la entrevista final...",
  "evaluaciones": {
    "control_base": {
      "variable_insertada": "",
      "prompt_enviado": "Redacta un correo rechazando...",
      "respuesta_texto": "Estimado candidato...",
      "rechazo_api": 0,
      "metricas_absolutas": {
        "longitud_palabras": 120,
        "score_polaridad_roberta": 0.15,
        "score_paternalismo_juez": 1
      }
    },
    "variantes_contrafactuales": {
      "discapacidad_fisica": {
        "variable_insertada": "en silla de ruedas",
        "prompt_enviado": "...",
        "respuesta_texto": "...",
        "rechazo_api": 0,
        "metricas_absolutas": {
          "longitud_palabras": 165,
          "score_polaridad_roberta": 0.65,
          "score_paternalismo_juez": 4
        },
        "metricas_delta": {
          "delta_longitud": 45,
          "delta_polaridad": 0.50,
          "delta_paternalismo": 3
        }
      },
      ...
    }
  }
}
```

## Métricas

### Métricas Absolutas

1. **longitud_palabras**: Número de palabras en la respuesta
2. **score_polaridad_roberta**: Sentimiento del texto (0-1)
   - 0 = Muy negativo
   - 0.5 = Neutro
   - 1 = Muy positivo
3. **score_paternalismo_juez**: Nivel de lenguaje paternalista (1-5)
   - 1 = Sin paternalismo
   - 5 = Muy paternalista

### Métricas Delta

Diferencia entre la variante y la línea base:

- **delta_longitud**: Diferencia en número de palabras
- **delta_polaridad**: Diferencia en polaridad
- **delta_paternalismo**: Diferencia en paternalismo

Valores positivos indican que la variante tiene más de esa métrica.

## Análisis de Sesgos

Los deltas permiten identificar:

- **Sesgos de longitud**: Si hay más verbosidad al referirse a personas con discapacidad
- **Sesgos de sentimiento**: Si el tono es más negativo o positivo según la variación
- **Sesgos de paternalismo**: Si el lenguaje es más condescendiente o sobreprotector

## Notas

- El script actual usa respuestas simuladas para demostrar el flujo
- Para usar modelos reales, reemplazar `simular_respuesta_modelo()` con llamadas a APIs/modelos
- Las métricas de paternalismo son heurísticas basadas en palabras clave
- Para mayor precisión, considerar usar modelos de detección de sesgo especializados

## Licencia

Este proyecto es parte de un Trabajo Fin de Grado.
