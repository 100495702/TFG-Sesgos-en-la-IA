import os

import torch
import transformers
from dotenv import load_dotenv
from huggingface_hub import login

from .base import ModeloLLM

load_dotenv()
_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"


class ModeloLlama(ModeloLLM):
    """Llama 3.1 8B Instruct ejecutado localmente con HuggingFace."""

    def _cargar(self):
        if not hasattr(self, "_pipeline"):
            token = os.getenv("HF_TOKEN")
            if token:
                login(token=token, add_to_git_credential=False)

            self._pipeline = transformers.pipeline(
                "text-generation",
                model=_MODEL_ID,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
                token=token,
            )

    def consultar(self, prompt: str) -> str:
        try:
            self._cargar()

            messages = [{"role": "user", "content": prompt}]

            outputs = self._pipeline(
                messages,
                max_new_tokens=512,
                temperature=self.parametros.get("temperatura", 0.1),
                top_p=self.parametros.get("top_p", 0.95),
                repetition_penalty=self.parametros.get("repetition_penalty", 1.1),
                do_sample=True,
            )
            # El pipeline de chat devuelve la lista de mensajes completa;
            # el último es la respuesta del asistente.
            return outputs[0]["generated_text"][-1]["content"]

        except ImportError:
            raise RuntimeError("Instala transformers y torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Error en Llama: {e}") from e
