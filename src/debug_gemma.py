"""
Debug-Skript für Gemma 2/3/4 – prüft ob das Modell lädt und generiert.
Kompatibel mit den unterschiedlichen Chat-Template Rückgabeformaten.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import GEMMA_MODEL_PATH

print("=" * 60)
print("GEMMA DEBUG TEST")
print("=" * 60)

# [1] Info
print(f"\n[1] Modell-Pfad: {GEMMA_MODEL_PATH}")
print(f"    CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")
    print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# [2] Tokenizer
print("\n[2] Lade Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"    Tokenizer-Typ: {type(tokenizer).__name__}")
print(f"    EOS-Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"    PAD-Token: {tokenizer.pad_token}")
print(f"    Chat-Template vorhanden: {tokenizer.chat_template is not None}")

# [3] Modell
print("\n[3] Lade Modell (kann dauern)...")
model = AutoModelForCausalLM.from_pretrained(
    GEMMA_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    local_files_only=True,
)
model.eval()
print(f"    Modell-Typ: {type(model).__name__}")
print(f"    Config: {model.config.model_type}")


def apply_chat_robust(prompt_text: str):
    """Wendet Chat-Template an, robust für Gemma 2 (Tensor) und Gemma 3 (dict)."""
    msgs = [{"role": "user", "content": prompt_text}]
    result = tokenizer.apply_chat_template(
        msgs, return_tensors="pt", add_generation_prompt=True,
        return_dict=True,
    )
    if hasattr(result, "data") or isinstance(result, dict):
        ids = result["input_ids"].to("cuda")
        mask = result.get("attention_mask", torch.ones_like(ids)).to("cuda")
    else:
        ids = result.to("cuda")
        mask = torch.ones_like(ids).to("cuda")
    return ids, mask


# [4] Einfacher Test ohne Chat-Template
print("\n[4] Test 1: Minimaler Prompt ohne Chat-Template")
print("-" * 60)
prompt = "Die Hauptstadt von Deutschland ist"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print(f"    Input-Tokens: {inputs['input_ids'].shape}")
print(f"    Input-Text: '{prompt}'")

with torch.no_grad():
    output = model.generate(
        **inputs, max_new_tokens=20, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

new_tokens = output[0][inputs['input_ids'].shape[1]:]
new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(f"    NEU GENERIERT: '{new_text}'")
print(f"    NEU-Token-Anzahl: {len(new_tokens)}")

# [5] Chat-Template Test
print("\n[5] Test 2: Mit Chat-Template")
print("-" * 60)
if tokenizer.chat_template is not None:
    ids, mask = apply_chat_robust("Sag 'Hallo Welt' auf Deutsch.")
    print(f"    Chat-Input-Tokens: {ids.shape}")

    with torch.no_grad():
        output = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=50, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][ids.shape[-1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"    NEU GENERIERT: '{new_text}'")
    print(f"    NEU-Token-Anzahl: {len(new_tokens)}")
else:
    print("    Kein Chat-Template vorhanden!")

# [6] JSON-Liste Test
print("\n[6] Test 3: JSON-Array mit 5 Obstsorten")
print("-" * 60)
if tokenizer.chat_template is not None:
    ids, mask = apply_chat_robust(
        "Nenne mir 5 Obstsorten als JSON-Array. Nur das Array, nichts anderes."
    )

    with torch.no_grad():
        output = model.generate(
            input_ids=ids, attention_mask=mask,
            max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output[0][ids.shape[-1]:]
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"    NEU GENERIERT:\n{new_text}")
    print(f"\n    Länge: {len(new_text)} Zeichen")

print("\n" + "=" * 60)
print("FERTIG. Alles OK wenn Test 2 und 3 sinnvolle Antworten geben.")
print("=" * 60)
