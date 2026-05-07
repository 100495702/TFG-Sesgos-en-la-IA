from transformers import pipeline

_MODEL_ID = "pysentimiento/robertuito-sentiment-analysis"


class AnalizadorSentimiento:
    """
    Análisis de sentimiento en español con robertuito (RoBERTa entrenado en tweets).
    Devuelve probabilidades POS/NEG/NEU y un score continuo = POS - NEG ∈ [-1, 1].
    """

    def __init__(self):
        self._pipe = None

    def _cargar(self):
        if self._pipe is None:
            self._pipe = pipeline(
                "text-classification",
                model=_MODEL_ID,
                top_k=None,
            )

    def analizar(self, texto: str) -> dict:
        if not texto or not texto.strip():
            return {"POS": 0.0, "NEG": 0.0, "NEU": 1.0, "score": 0.0}

        self._cargar()
        texto_truncado = " ".join(texto.split()[:200])
        resultados = self._pipe(texto_truncado)[0]
        probs = {r["label"]: round(r["score"], 4) for r in resultados}
        probs["score"] = round(probs.get("POS", 0.0) - probs.get("NEG", 0.0), 4)
        return probs
