import os

from .base import ModeloLLM
from .claude import ModeloClaude
from .gemini import ModeloGemini
from .llama import ModeloLlama
from .rigochat import ModeloRigoChat

_REGISTRO: dict = {
    "Llama_3_8B_Instruct": lambda p: ModeloLlama(p),
    "RigoChat_7B_v2": lambda p: ModeloRigoChat(p),
    "Gemini_1.5_Flash": lambda p: ModeloGemini(p, os.getenv("GOOGLE_API_KEY", "")),
    "Claude_3.5_Sonnet": lambda p: ModeloClaude(p, os.getenv("ANTHROPIC_API_KEY", "")),
}


def crear_modelo(nombre: str, parametros: dict) -> ModeloLLM:
    """Devuelve la instancia del modelo correspondiente al nombre del dataset."""
    if nombre not in _REGISTRO:
        raise ValueError(f"Modelo no reconocido: '{nombre}'. Disponibles: {list(_REGISTRO)}")
    return _REGISTRO[nombre](parametros)
