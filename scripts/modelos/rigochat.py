import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModeloLLM

load_dotenv()
_MODEL_NAME = "IIC/RigoChat-7b-v2"


class ModeloRigoChat(ModeloLLM):
    """RigoChat 7B v2 ejecutado localmente con HuggingFace."""

    def _cargar(self):
        if not hasattr(self, "_model"):
            token = os.getenv("HF_TOKEN")
            if token:
                login(token=token, add_to_git_credential=False)

            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_NAME,
                token=token,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                _MODEL_NAME,
                token=token,
                dtype=torch.bfloat16,
                device_map="auto",  # distribuye entre todas las GPUs disponibles
                trust_remote_code=True,
            )
            self._device = next(self._model.parameters()).device

    def consultar(self, prompt: str) -> str:
        try:
            self._cargar()

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.parametros.get("temperatura", 0.1),
                    top_p=self.parametros.get("top_p", 0.95),
                    repetition_penalty=self.parametros.get("repetition_penalty", 1.15),
                    do_sample=True,
                )

            # Decodificar solo los tokens generados, sin el prompt de entrada
            tokens_generados = outputs[0][inputs["input_ids"].shape[1]:]
            return self._tokenizer.decode(tokens_generados, skip_special_tokens=True)

        except ImportError:
            raise RuntimeError("Instala transformers y torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Error en RigoChat: {e}") from e
