"""
SCHRITT 1: Taxonomie generieren
─────────────────────────────────
Gemma analysiert ein Sample der Meldungen und schlägt eine Taxonomie
von ~70 Kategorien vor (Vorfall-spezifisch).

Output:
  cache/categories.json  – Liste der Kategorien mit Beschreibungen
"""
import json
import logging
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CACHE_DIR, COL_ISSUE, N_CATEGORIES_TARGET,
    SAMPLE_SIZE_FOR_TAXONOMY, RANDOM_SEED
)
from data_utils import load_and_preprocess
from gemma_wrapper import get_gemma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


TAXONOMY_PROMPT = """Du bist Experte für die Analyse von polizeilichen Meldungen und Kriminalitätsdaten.

Analysiere die folgenden {n_samples} Beispiel-Meldungen und erstelle eine saubere Taxonomie
von genau {n_cat} Kategorien, in die sich ALLE Meldungen einordnen lassen.

Anforderungen an die Kategorien:
- Kurz und präzise (1-3 Wörter)
- Trennscharf (keine Überlappungen)
- Decken das gesamte Spektrum der Meldungen ab
- Auf Deutsch
- Sinnvolle Granularität (nicht zu grob, nicht zu fein)

BEISPIEL-MELDUNGEN:
{samples}

Gib die Kategorien als JSON-Array zurück, jede Kategorie mit "name" und "beschreibung".
NUR das JSON-Array, keine Einleitung, kein Abschluss-Text.

Beispiel-Format:
[
  {{"name": "Sachbeschädigung", "beschreibung": "Beschädigung fremden Eigentums"}},
  {{"name": "Ruhestörung", "beschreibung": "Lärm, laute Musik, nächtliche Störungen"}}
]

Deine Antwort:"""


def build_sample_text(texts: list, max_chars: int = 300) -> str:
    """Formatiert Sample-Texte für den Prompt."""
    lines = []
    for i, t in enumerate(texts, 1):
        t = str(t).strip().replace("\n", " ")
        if len(t) > max_chars:
            t = t[:max_chars] + "..."
        lines.append(f"{i}. {t}")
    return "\n".join(lines)


def parse_categories(response: str) -> list:
    """Extrahiert JSON-Liste aus Gemma-Antwort, robust gegen Zusatz-Text."""
    # JSON-Array finden
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"Kein JSON-Array in Antwort gefunden:\n{response[:500]}")

    json_str = response[start:end + 1]
    try:
        categories = json.loads(json_str)
    except json.JSONDecodeError as e:
        # Versuch, häufige Fehler zu reparieren
        json_str = json_str.replace(",]", "]").replace(",}", "}")
        categories = json.loads(json_str)

    # Validierung
    if not isinstance(categories, list):
        raise ValueError("Antwort ist keine Liste")

    # Auf einheitliches Format bringen
    cleaned = []
    for c in categories:
        if isinstance(c, dict) and "name" in c:
            cleaned.append({
                "name": str(c["name"]).strip(),
                "beschreibung": str(c.get("beschreibung", "")).strip(),
            })
        elif isinstance(c, str):
            cleaned.append({"name": c.strip(), "beschreibung": ""})

    return cleaned


def main():
    logger.info("=" * 60)
    logger.info("SCHRITT 1: Kategorie-Taxonomie generieren")
    logger.info("=" * 60)

    # Daten laden
    df = load_and_preprocess()
    logger.info(f"Geladene Einträge: {len(df)}")

    # Zufalls-Sample ziehen
    random.seed(RANDOM_SEED)
    sample_size = min(SAMPLE_SIZE_FOR_TAXONOMY, len(df))
    sample_texts = df[COL_ISSUE].sample(sample_size, random_state=RANDOM_SEED).tolist()
    logger.info(f"Sample-Größe: {sample_size}")

    # Gemma aufrufen
    gemma = get_gemma()
    sample_block = build_sample_text(sample_texts)
    prompt = TAXONOMY_PROMPT.format(
        n_samples=sample_size,
        n_cat=N_CATEGORIES_TARGET,
        samples=sample_block,
    )

    logger.info("Gemma generiert Taxonomie...")
    response = gemma.generate(prompt, max_new_tokens=4096, temperature=0.3)

    # Ergebnis parsen
    try:
        categories = parse_categories(response)
    except Exception as e:
        logger.error(f"Parsing fehlgeschlagen: {e}")
        # Rohantwort sichern für manuelle Inspektion
        raw_path = CACHE_DIR / "taxonomy_raw_response.txt"
        raw_path.write_text(response, encoding="utf-8")
        logger.error(f"Rohantwort gespeichert unter: {raw_path}")
        raise

    logger.info(f"{len(categories)} Kategorien erhalten.")

    # Speichern
    out_path = CACHE_DIR / "categories.json"
    out_path.write_text(
        json.dumps(categories, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    logger.info(f"Gespeichert: {out_path}")

    # Übersicht anzeigen
    print("\n" + "=" * 60)
    print("GENERIERTE KATEGORIEN:")
    print("=" * 60)
    for i, c in enumerate(categories, 1):
        desc = f" – {c['beschreibung']}" if c.get("beschreibung") else ""
        print(f"  {i:3d}. {c['name']}{desc}")
    print("\nTipp: Du kannst die Datei cache/categories.json manuell editieren,")
    print("      bevor du Schritt 2 startest.")


if __name__ == "__main__":
    main()
