import os

from .base import ModeloLLM
from .gemini import ModeloGemini
from .llama import ModeloLlama
from .rigochat import ModeloRigoChat
from .salamandra import ModeloSalamandra

_REGISTRO: dict = {
    "Llama_3.1_8B_Instruct": lambda p: ModeloLlama(p),
    "RigoChat_7B_v2": lambda p: ModeloRigoChat(p),
    "Gemini_2.5_Flash": lambda p: ModeloGemini(p, os.getenv("GOOGLE_API_KEY", "")),
    "Salamandra_7B_Instruct": lambda p: ModeloSalamandra(p),
}


def crear_modelo(nombre: str, parametros: dict) -> ModeloLLM:
    """Devuelve la instancia del modelo correspondiente al nombre del dataset."""
    if nombre not in _REGISTRO:
        raise ValueError(f"Modelo no reconocido: '{nombre}'. Disponibles: {list(_REGISTRO)}")
    return _REGISTRO[nombre](parametros)
