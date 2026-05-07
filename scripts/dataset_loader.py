import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


class DatasetLoader:
    """Carga dataset.json y expone los datos del experimento."""

    def __init__(self, ruta: str):
        self.ruta = Path(ruta)
        with open(self.ruta, encoding="utf-8") as f:
            datos = json.load(f)

        self.prompts: List[Dict[str, Any]] = datos["dataset"]
        self.config_modelos: Dict[str, Any] = datos.get("configuracion_modelos", {})

    # ---- Modelos ----

    def nombres_modelos(self) -> List[str]:
        return list(self.config_modelos.keys())

    def parametros_modelo(self, nombre: str) -> Dict[str, Any]:
        return self.config_modelos.get(nombre, {}).get("parametros", {})

    # ---- Variaciones ----

    def variacion_base(self, prompt: Dict) -> Dict:
        """Devuelve la variación neutral (sin discapacidad)."""
        return next(v for v in prompt["variaciones"] if v["condicion"] == "neutral")

    def variaciones_discapacidad(self, prompt: Dict) -> List[Dict]:
        """Devuelve las variaciones con discapacidad."""
        return [v for v in prompt["variaciones"] if v["condicion"] == "discapacidad"]

    # ---- Iteración ----

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[Dict]:
        return iter(self.prompts)
