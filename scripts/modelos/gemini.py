import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .base import ModeloLLM

load_dotenv()
_MODEL_ID = "gemini-2.5-flash"


class ModeloGemini(ModeloLLM):
    """Gemini 2.0 Flash vía Google Gemini API."""

    def __init__(self, parametros: dict, api_key: str):
        super().__init__(parametros)
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no configurada")
        self._client = genai.Client(api_key=api_key)

    def consultar(self, prompt: str) -> str:
        try:
            respuesta = self._client.models.generate_content(
                model=_MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.parametros.get("temperatura", 0.1),
                    top_p=self.parametros.get("top_p", 0.95),
                    max_output_tokens=8192,
                ),
            )
            return respuesta.text

        except Exception as e:
            raise RuntimeError(f"Error en Gemini: {e}") from e
