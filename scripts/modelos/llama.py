from .base import ModeloLLM


class ModeloLlama(ModeloLLM):
    """Llama 3 8B Instruct ejecutado localmente con HuggingFace."""

    def consultar(self, prompt: str) -> str:
        try:
            from transformers import pipeline

            if not hasattr(self, "_pipeline"):
                self._pipeline = pipeline(
                    "text-generation",
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    device=0,
                )

            salida = self._pipeline(
                prompt,
                max_new_tokens=512,
                temperature=self.parametros.get("temperatura", 0.1),
                top_p=self.parametros.get("top_p", 0.95),
                repetition_penalty=self.parametros.get("repetition_penalty", 1.1),
                do_sample=True,
            )
            return salida[0]["generated_text"]

        except ImportError:
            raise RuntimeError("Instala transformers y torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Error en Llama: {e}") from e
