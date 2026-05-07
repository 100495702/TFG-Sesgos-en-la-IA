from .base import ModeloLLM


class ModeloClaude(ModeloLLM):
    """Claude 3.5 Sonnet vía Anthropic API."""

    def __init__(self, parametros: dict, api_key: str):
        super().__init__(parametros)
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY no configurada")
        self.api_key = api_key

    def consultar(self, prompt: str) -> str:
        try:
            from anthropic import Anthropic

            cliente = Anthropic(api_key=self.api_key)
            respuesta = cliente.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=512,
                temperature=self.parametros.get("temperatura", 0.1),
                top_p=self.parametros.get("top_p", 0.95),
                messages=[{"role": "user", "content": prompt}],
            )
            return respuesta.content[0].text

        except ImportError:
            raise RuntimeError("Instala anthropic: pip install anthropic")
        except Exception as e:
            raise RuntimeError(f"Error en Claude: {e}") from e
