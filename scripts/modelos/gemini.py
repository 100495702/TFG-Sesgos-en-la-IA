from .base import ModeloLLM


class ModeloGemini(ModeloLLM):
    """Gemini 1.5 Flash vía Google Generative AI API."""

    def __init__(self, parametros: dict, api_key: str):
        super().__init__(parametros)
        if not api_key:
            raise ValueError("GOOGLE_API_KEY no configurada")
        self.api_key = api_key

    def consultar(self, prompt: str) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            modelo = genai.GenerativeModel("gemini-1.5-flash")

            respuesta = modelo.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.parametros.get("temperatura", 0.1),
                    top_p=self.parametros.get("top_p", 0.95),
                    max_output_tokens=512,
                ),
            )
            return respuesta.text

        except ImportError:
            raise RuntimeError("Instala google-generativeai: pip install google-generativeai")
        except Exception as e:
            raise RuntimeError(f"Error en Gemini: {e}") from e
