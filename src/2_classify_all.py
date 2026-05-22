"""
SCHRITT 2: Alle Meldungen klassifizieren
─────────────────────────────────────────
Gemma ordnet jeden Text einer der Kategorien aus Schritt 1 zu.
Nutzt Batch-Processing und Caching (bereits klassifizierte Texte werden übersprungen).

Output:
  cache/category_assignments.csv  – Alle Texte mit zugewiesener Kategorie
"""
import json
import logging
import hashlib
from pathlib import Path
import pandas as pd
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE_DIR, COL_ISSUE, BATCH_SIZE_CLASSIFY
from data_utils import load_and_preprocess
from gemma_wrapper import get_gemma

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


CLASSIFY_PROMPT = """Du bist Experte für die Klassifikation von polizeilichen Meldungen.

Ordne die folgende Meldung GENAU EINER der vorgegebenen Kategorien zu.
Antworte NUR mit dem exakten Kategorie-Namen aus der Liste – nichts anderes.

KATEGORIEN:
{categories}

MELDUNG:
{text}

Kategorie:"""


def text_hash(text: str) -> str:
    """Stabiler Hash eines Textes für Caching."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def normalize_category(response: str, valid_categories: list) -> str:
    """
    Matcht Gemma-Antwort gegen gültige Kategorien.
    Fallback auf "Sonstiges" wenn keine Übereinstimmung.
    """
    response = response.strip().split("\n")[0].strip()
    response = response.strip(".,;:!?\"'")

    # Exakter Match
    for cat in valid_categories:
        if response.lower() == cat.lower():
            return cat

    # Enthält die Kategorie?
    for cat in valid_categories:
        if cat.lower() in response.lower():
            return cat

    # Fallback
    return "Sonstiges"


def build_category_list(categories: list) -> str:
    lines = []
    for c in categories:
        if c.get("beschreibung"):
            lines.append(f"- {c['name']}: {c['beschreibung']}")
        else:
            lines.append(f"- {c['name']}")
    return "\n".join(lines)


def main():
    logger.info("=" * 60)
    logger.info("SCHRITT 2: Texte klassifizieren")
    logger.info("=" * 60)

    # Kategorien laden
    cat_path = CACHE_DIR / "categories.json"
    if not cat_path.exists():
        raise FileNotFoundError(
            f"Kategorien-Datei fehlt: {cat_path}\n"
            f"→ Bitte zuerst '1_extract_categories.py' ausführen."
        )
    categories = json.loads(cat_path.read_text(encoding="utf-8"))
    # Sicherstellen dass "Sonstiges" vorhanden ist
    if not any(c["name"].lower() == "sonstiges" for c in categories):
        categories.append({"name": "Sonstiges", "beschreibung": "Nicht eindeutig zuordenbar"})
    valid_names = [c["name"] for c in categories]
    cat_list_str = build_category_list(categories)
    logger.info(f"Kategorien geladen: {len(categories)}")

    # Daten laden
    df = load_and_preprocess()
    df["text_hash"] = df[COL_ISSUE].apply(text_hash)
    logger.info(f"Zu klassifizieren: {len(df)} Meldungen")

    # Vorhandene Zuordnungen laden (Caching)
    cache_path = CACHE_DIR / "category_assignments.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_hashes = dict(zip(cached["text_hash"], cached["category"]))
        logger.info(f"Bereits klassifiziert (Cache): {len(cached_hashes)}")
    else:
        cached_hashes = {}

    # Zu klassifizierende Texte ermitteln
    unique_texts = df.drop_duplicates(subset="text_hash")
    to_classify = unique_texts[~unique_texts["text_hash"].isin(cached_hashes)]
    logger.info(f"Neu zu klassifizieren: {len(to_classify)}")

    if len(to_classify) == 0:
        logger.info("Alles bereits im Cache. Erstelle finale Zuordnungs-Datei...")
    else:
        # Gemma laden
        gemma = get_gemma()
        gemma.load()

        # Batch-Processing
        texts = to_classify[COL_ISSUE].tolist()
        hashes = to_classify["text_hash"].tolist()

        batch = BATCH_SIZE_CLASSIFY
        pbar = tqdm(range(0, len(texts), batch), desc="Klassifiziere")
        for start in pbar:
            chunk = texts[start:start + batch]
            hash_chunk = hashes[start:start + batch]

            prompts = [
                CLASSIFY_PROMPT.format(categories=cat_list_str, text=t[:1000])
                for t in chunk
            ]

            try:
                responses = gemma.generate_batch(
                    prompts, max_new_tokens=20, temperature=0.0
                )
            except Exception as e:
                logger.warning(f"Batch-Fehler: {e}. Fallback auf Einzelverarbeitung.")
                responses = []
                for p in prompts:
                    try:
                        r = gemma.generate(p, max_new_tokens=20, temperature=0.0)
                        responses.append(r)
                    except Exception:
                        responses.append("Sonstiges")

            for h, r in zip(hash_chunk, responses):
                cached_hashes[h] = normalize_category(r, valid_names)

            # Zwischen-Speichern alle 500 Texte
            if (start // batch) % 50 == 0 and start > 0:
                _save_cache(cached_hashes, cache_path)

        # Final speichern
        _save_cache(cached_hashes, cache_path)

    # Vollständige Zuordnungstabelle erstellen
    df["category"] = df["text_hash"].map(cached_hashes).fillna("Sonstiges")
    out_path = CACHE_DIR / "classified_data.csv"
    df[["datetime", "hour", "weekday", "month", "is_weekend",
        "dayofyear", "time_confidence", "place",
        COL_ISSUE, "category", "text_hash"]].to_csv(
        out_path, index=False, encoding="utf-8-sig"
    )
    logger.info(f"Vollständige Zuordnungen gespeichert: {out_path}")

    # Statistik
    print("\n" + "=" * 60)
    print("KATEGORIE-VERTEILUNG:")
    print("=" * 60)
    counts = df["category"].value_counts()
    for cat, n in counts.head(30).items():
        pct = 100 * n / len(df)
        bar = "█" * int(pct)
        print(f"  {cat:30s} {n:6d}  ({pct:5.1f}%) {bar}")
    if len(counts) > 30:
        print(f"  ... und {len(counts) - 30} weitere Kategorien")


def _save_cache(cache_dict: dict, path: Path):
    """Speichert das Hash→Kategorie Cache als CSV."""
    cache_df = pd.DataFrame(
        [(h, c) for h, c in cache_dict.items()],
        columns=["text_hash", "category"]
    )
    cache_df.to_csv(path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
