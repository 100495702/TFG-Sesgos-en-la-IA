"""
Pipeline principal: carga dataset.json, genera respuestas y calcula métricas.
Salida: outputs/salidas.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dataset_loader import DatasetLoader
from scripts.evaluacion import AnalizadorSentimiento, JuezLLM
from scripts.modelos import ModeloLLM, crear_modelo


class GeneradorEvaluaciones:

    def __init__(self, dataset_path: str):
        self.output_path = PROJECT_ROOT / "outputs" / "salidas.json"

        self.loader = DatasetLoader(dataset_path)
        self.dataset = self.loader.prompts

        self.sentimiento = AnalizadorSentimiento()
        self.juez = JuezLLM()
        self._modelos_cache: Dict[str, ModeloLLM] = {}
        self.resultados = []

    def _obtener_modelo(self, nombre: str) -> ModeloLLM:
        if nombre not in self._modelos_cache:
            self._modelos_cache[nombre] = crear_modelo(
                nombre, self.loader.parametros_modelo(nombre)
            )
        return self._modelos_cache[nombre]

    def procesar_prueba(self, prueba: Dict[str, Any], modelo: str, num_prueba: int) -> Dict[str, Any]:
        base = self.loader.variacion_base(prueba)

        resultado = {
            "id_prueba": f"{num_prueba:03d}",
            "id_prompt_original": prueba["id_prompt"],
            "dominio": prueba["dominio"],
            "objetivo_sesgo": prueba["objetivo_sesgo"],
            "modelo_evaluado": modelo,
            "plantilla_prompt": prueba["plantilla"],
            "evaluaciones": {
                "control_base": self._procesar_variante(base["prompt_final"], modelo),
                "variantes_contrafactuales": {},
            },
        }

        for var in self.loader.variaciones_discapacidad(prueba):
            tipo = var["tipo_discapacidad"]
            resultado["evaluaciones"]["variantes_contrafactuales"][tipo] = \
                self._procesar_variante(var["prompt_final"], modelo)

        return resultado

    def _procesar_variante(self, prompt_final: str, nombre_modelo: str) -> Dict[str, Any]:
        rechazo_api = 0
        try:
            respuesta_texto = self._obtener_modelo(nombre_modelo).consultar(prompt_final)
        except Exception as e:
            print(f"    [!] Error al consultar {nombre_modelo}: {e}")
            respuesta_texto = ""
            rechazo_api = 1

        sent = self.sentimiento.analizar(respuesta_texto)
        juicio = self.juez.evaluar(prompt_final, respuesta_texto)

        return {
            "prompt_enviado": prompt_final,
            "respuesta_texto": respuesta_texto,
            "rechazo_api": rechazo_api,
            "metricas_absolutas": {
                "longitud_palabras": len(respuesta_texto.split()),
                "sentimiento": sent,
                "juez_llm": juicio,
            },
        }

    def calcular_deltas(self, resultado: Dict[str, Any]) -> Dict[str, Any]:
        base = resultado["evaluaciones"]["control_base"]["metricas_absolutas"]

        for variante in resultado["evaluaciones"]["variantes_contrafactuales"].values():
            m = variante["metricas_absolutas"]
            variante["metricas_delta"] = {
                "delta_longitud": m["longitud_palabras"] - base["longitud_palabras"],
                "delta_sentimiento": round(
                    m["sentimiento"]["score"] - base["sentimiento"]["score"], 4
                ),
                "delta_juez": {
                    dim: m["juez_llm"].get(dim, 0) - base["juez_llm"].get(dim, 0)
                    for dim in ("relevancia", "exactitud", "utilidad",
                                "tono_respetuoso", "ausencia_estereotipo")
                },
            }

        return resultado

    def generar_evaluaciones(
        self,
        modelos: List[str] = None,
        limite_prompts: int = None,
    ) -> List[Dict]:
        if modelos is None:
            modelos = self.loader.nombres_modelos()

        print(f"Modelos: {modelos}")
        print(f"Prompts: {len(self.dataset)}" + (f" (límite: {limite_prompts})" if limite_prompts else ""))

        num_prueba = 1
        for idx, prueba in enumerate(self.dataset):
            if limite_prompts and idx >= limite_prompts:
                break

            print(f"\n[{idx+1}/{len(self.dataset)}] {prueba['id_prompt']} — {prueba['dominio']}")

            for modelo in modelos:
                print(f"  → {modelo}")
                resultado = self.procesar_prueba(prueba, modelo, num_prueba)
                resultado = self.calcular_deltas(resultado)
                self.resultados.append(resultado)
                num_prueba += 1

        print(f"\nTotal evaluaciones generadas: {len(self.resultados)}")
        return self.resultados

    def guardar_resultados(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.resultados, f, ensure_ascii=False, indent=2)
        print(f"Resultados guardados en: {self.output_path}")


def main():
    dataset_path = PROJECT_ROOT / "data" / "dataset.json"
    if not dataset_path.exists():
        print(f"Error: no se encontró {dataset_path}")
        sys.exit(1)

    generador = GeneradorEvaluaciones(str(dataset_path))
    generador.generar_evaluaciones(limite_prompts=2)
    generador.guardar_resultados()


if __name__ == "__main__":
    main()
