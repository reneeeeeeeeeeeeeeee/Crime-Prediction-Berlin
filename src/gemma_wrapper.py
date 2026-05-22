"""
Gemma Wrapper - kompatibel mit Gemma 2 UND Gemma 3.
Gemma 3 gibt beim apply_chat_template ein Dict zurück (input_ids + attention_mask),
Gemma 2 einen einzelnen Tensor. Der Wrapper behandelt beide Fälle.
"""
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from typing import List
from config import GEMMA_MODEL_PATH, GEMMA_DEVICE, GEMMA_DTYPE

logger = logging.getLogger(__name__)


class GemmaWrapper:
    """Einfacher Wrapper um das lokale Gemma-Modell."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path or GEMMA_MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.device = GEMMA_DEVICE if torch.cuda.is_available() else "cpu"
        self._dtype = torch.bfloat16 if GEMMA_DTYPE == "bfloat16" else torch.float16

    def load(self):
        """Modell laden (einmalig)."""
        if self.model is not None:
            return

        logger.info(f"Lade Gemma von: {self.model_path}")
        logger.info(f"Device: {self.device} | DType: {self._dtype}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self._dtype if self.device == "cuda" else torch.float32,
            device_map=self.device,
            local_files_only=True,
        )
        self.model.eval()

        # PAD-Token auf EOS, falls nicht gesetzt
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info("Gemma erfolgreich geladen.")

    def _apply_chat(self, prompt: str, return_dict: bool = True):
        """
        Wendet das Chat-Template an und gibt einen Dict mit input_ids + attention_mask zurück.
        Funktioniert sowohl mit Gemma 2 (Tensor) als auch Gemma 3 (BatchEncoding).
        """
        messages = [{"role": "user", "content": prompt}]
        result = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=return_dict,
        )

        # Gemma 3: BatchEncoding / dict-artig → direkt nutzbar
        if hasattr(result, "data") or isinstance(result, dict):
            input_ids = result["input_ids"]
            attention_mask = result.get("attention_mask",
                                        torch.ones_like(input_ids))
        else:
            # Gemma 2 und früher: direkt ein Tensor
            input_ids = result
            attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
        }

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.3) -> str:
        """Einzelner Generate-Aufruf."""
        self.load()

        has_template = (hasattr(self.tokenizer, "apply_chat_template")
                        and self.tokenizer.chat_template is not None)

        if has_template:
            inputs = self._apply_chat(prompt)
        else:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            inputs = {"input_ids": enc["input_ids"],
                      "attention_mask": enc.get("attention_mask",
                                                 torch.ones_like(enc["input_ids"]))}

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01),
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,
            )

        new_tokens = output[0][input_len:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 50,
                       temperature: float = 0.1) -> List[str]:
        """Batch-Verarbeitung für schnellere Klassifikation."""
        self.load()

        has_template = (hasattr(self.tokenizer, "apply_chat_template")
                        and self.tokenizer.chat_template is not None)

        # Alle Prompts einzeln mit Chat-Template formatieren (ohne tokenize)
        if has_template:
            formatted = []
            for p in prompts:
                msg = [{"role": "user", "content": p}]
                s = self.tokenizer.apply_chat_template(
                    msg, tokenize=False, add_generation_prompt=True
                )
                formatted.append(s)
        else:
            formatted = prompts

        # Padding links für Batch-Generation
        self.tokenizer.padding_side = "left"

        enc = self.tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True,
            max_length=2048,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=max(temperature, 0.01),
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,
            )

        results = []
        input_len = input_ids.shape[1]
        for i in range(output.shape[0]):
            new_tokens = output[i][input_len:]
            decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(decoded.strip())

        return results

    def unload(self):
        """Speicher freigeben."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Gemma entladen.")


# Singleton
_gemma_instance = None

def get_gemma() -> GemmaWrapper:
    global _gemma_instance
    if _gemma_instance is None:
        _gemma_instance = GemmaWrapper()
    return _gemma_instance
