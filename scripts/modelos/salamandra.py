import os
from datetime import datetime

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import ModeloLLM

load_dotenv()
_MODEL_ID = "BSC-LT/salamandra-7b-instruct"


class ModeloSalamandra(ModeloLLM):
    """Salamandra 7B Instruct (BSC-LT) ejecutado localmente con HuggingFace."""

    def _cargar(self):
        if not hasattr(self, "_model"):
            token = os.getenv("HF_TOKEN")
            if token:
                login(token=token, add_to_git_credential=False)

            self._tokenizer = AutoTokenizer.from_pretrained(
                _MODEL_ID,
                token=token,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID,
                device_map="auto",
                dtype=torch.bfloat16,
                token=token,
            )

    def consultar(self, prompt: str) -> str:
        try:
            self._cargar()

            message = [{"role": "user", "content": prompt}]
            date_string = datetime.today().strftime("%Y-%m-%d")

            prompt_formateado = self._tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                date_string=date_string,
            )

            inputs = self._tokenizer.encode(
                prompt_formateado,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=inputs,
                    max_new_tokens=512,
                    temperature=self.parametros.get("temperatura", 0.1),
                    top_p=self.parametros.get("top_p", 0.95),
                    repetition_penalty=self.parametros.get("repetition_penalty", 1.1),
                    do_sample=True,
                )

            # Decodificar solo los tokens generados, sin el prompt de entrada
            tokens_generados = outputs[0][inputs.shape[1]:]
            return self._tokenizer.decode(tokens_generados, skip_special_tokens=True)

        except ImportError:
            raise RuntimeError("Instala transformers y torch: pip install transformers torch")
        except Exception as e:
            raise RuntimeError(f"Error en Salamandra: {e}") from e
