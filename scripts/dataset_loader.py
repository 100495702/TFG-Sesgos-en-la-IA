import json
from pathlib import Path
from typing import Any, Dict, Iterator, List


_CAMPOS_PROMPT = {"id_prompt", "dominio", "plantilla", "variaciones"}
_CAMPOS_VARIANTE = {"condicion", "tipo_discapacidad", "texto_condicion", "prompt_final"}


class DatasetLoader:
    """Carga dataset.json y expone los datos del experimento."""

    def __init__(self, ruta: str):
        self.ruta = Path(ruta)
        with open(self.ruta, encoding="utf-8") as f:
            datos = json.load(f)

        self.prompts: List[Dict[str, Any]] = datos["dataset"]
        self.config_modelos: Dict[str, Any] = datos.get("configuracion_modelos", {})
        self._validar()

    def _validar(self) -> None:
        """RF2: valida que cada prompt y variante contiene los campos mínimos requeridos."""
        for i, prompt in enumerate(self.prompts):
            faltantes = _CAMPOS_PROMPT - prompt.keys()
            if faltantes:
                raise ValueError(f"Prompt #{i} le faltan campos: {faltantes}")
            for j, var in enumerate(prompt.get("variaciones", [])):
                faltantes_var = _CAMPOS_VARIANTE - var.keys()
                if faltantes_var:
                    raise ValueError(
                        f"Prompt #{i} variante #{j} le faltan campos: {faltantes_var}"
                    )

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
