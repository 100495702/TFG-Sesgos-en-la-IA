"""
RF11 — Selección de muestra para evaluación manual.
Genera outputs/muestra_evaluacion_manual.json con N casos por criterio.

Uso:
    python scripts/seleccionar_muestra.py [--n 5] [--criterio sesgo|aleatorio|estratificado]
"""

import argparse
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SALIDAS = PROJECT_ROOT / "outputs" / "salidas.json"
OUTPUT = PROJECT_ROOT / "outputs" / "muestra_evaluacion_manual.json"


def _casos_extremos(data: list, n_por_modelo: int) -> list:
    """Selecciona los n casos con mayor delta negativo en ausencia_estereotipo por modelo."""
    modelos = list({r["modelo_evaluado"] for r in data})
    muestra = []
    for modelo in modelos:
        candidatos = []
        for r in data:
            if r["modelo_evaluado"] != modelo:
                continue
            for tipo, var in r["evaluaciones"]["variantes_contrafactuales"].items():
                if "metricas_delta" not in var:
                    continue
                score = var["metricas_delta"]["delta_juez"].get("ausencia_estereotipo", 0)
                candidatos.append({
                    "id_prueba": r["id_prueba"],
                    "id_prompt": r["id_prompt_original"],
                    "dominio": r["dominio"],
                    "modelo": modelo,
                    "tipo_discapacidad": tipo,
                    "delta_ausencia_estereotipo": score,
                    "prompt_enviado": var["prompt_enviado"],
                    "respuesta_texto": var["respuesta_texto"],
                    "metricas_absolutas": var["metricas_absolutas"],
                    "metricas_delta": var["metricas_delta"],
                })
        candidatos.sort(key=lambda x: x["delta_ausencia_estereotipo"])
        muestra.extend(candidatos[:n_por_modelo])
    return muestra


def _muestra_aleatoria(data: list, n: int) -> list:
    """Selecciona n casos al azar del conjunto completo."""
    casos = []
    for r in data:
        for tipo, var in r["evaluaciones"]["variantes_contrafactuales"].items():
            casos.append({
                "id_prueba": r["id_prueba"],
                "id_prompt": r["id_prompt_original"],
                "dominio": r["dominio"],
                "modelo": r["modelo_evaluado"],
                "tipo_discapacidad": tipo,
                "prompt_enviado": var["prompt_enviado"],
                "respuesta_texto": var["respuesta_texto"],
                "metricas_absolutas": var.get("metricas_absolutas", {}),
                "metricas_delta": var.get("metricas_delta", {}),
            })
    return random.sample(casos, min(n, len(casos)))


def _muestra_estratificada(data: list, n_por_celda: int) -> list:
    """Selecciona n casos por cada combinación (modelo, dominio)."""
    from collections import defaultdict
    celdas = defaultdict(list)
    for r in data:
        for tipo, var in r["evaluaciones"]["variantes_contrafactuales"].items():
            clave = (r["modelo_evaluado"], r["dominio"])
            celdas[clave].append({
                "id_prueba": r["id_prueba"],
                "id_prompt": r["id_prompt_original"],
                "dominio": r["dominio"],
                "modelo": r["modelo_evaluado"],
                "tipo_discapacidad": tipo,
                "prompt_enviado": var["prompt_enviado"],
                "respuesta_texto": var["respuesta_texto"],
                "metricas_absolutas": var.get("metricas_absolutas", {}),
                "metricas_delta": var.get("metricas_delta", {}),
            })
    muestra = []
    for casos in celdas.values():
        muestra.extend(random.sample(casos, min(n_por_celda, len(casos))))
    return muestra


def main():
    parser = argparse.ArgumentParser(description="Selecciona muestra para evaluación manual")
    parser.add_argument("--n", type=int, default=5, help="Casos por modelo/celda (default: 5)")
    parser.add_argument(
        "--criterio",
        choices=["sesgo", "aleatorio", "estratificado"],
        default="sesgo",
        help="Criterio de selección (default: sesgo — casos más extremos por modelo)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(SALIDAS, encoding="utf-8") as f:
        data = json.load(f)

    if args.criterio == "sesgo":
        muestra = _casos_extremos(data, args.n)
    elif args.criterio == "aleatorio":
        muestra = _muestra_aleatoria(data, args.n)
    else:
        muestra = _muestra_estratificada(data, args.n)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(muestra, f, ensure_ascii=False, indent=2)

    print(f"Muestra guardada en: {OUTPUT}")
    print(f"Total casos seleccionados: {len(muestra)}")
    print(f"Criterio: {args.criterio} | n={args.n} | seed={args.seed}")


if __name__ == "__main__":
    main()