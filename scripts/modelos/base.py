from abc import ABC, abstractmethod


class ModeloLLM(ABC):
    """Clase base para todos los modelos del experimento."""

    def __init__(self, parametros: dict):
        self.parametros = parametros

    @abstractmethod
    def consultar(self, prompt: str) -> str:
        """Envía el prompt al modelo y devuelve la respuesta en texto."""
        ...
