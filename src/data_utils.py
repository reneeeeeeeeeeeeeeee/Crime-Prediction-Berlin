"""
Utility-Modul: Excel laden und vorverarbeiten.

Das Datum in der 'date'-Spalte ist bereits korrekt (nicht Melde-Datum,
sondern Ereignis-Datum). Wir verschieben es NICHT. Aus der 'time'-Spalte
extrahieren wir nur die Stunde – egal ob exakt ("16:00") oder fuzzy
("gestern abend").
"""
import pandas as pd
from pathlib import Path
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, EXCEL_PATTERN, COL_DATE, COL_TIME, COL_ISSUE, COL_PLACE
from time_parser import parse_series

logger = logging.getLogger(__name__)


def load_all_excels() -> pd.DataFrame:
    """Lädt alle meldungen_YYYY.xlsx aus dem data-Ordner."""
    excel_files = sorted(DATA_DIR.glob(EXCEL_PATTERN))
    if not excel_files:
        raise FileNotFoundError(
            f"Keine Excel-Dateien gefunden in {DATA_DIR} (Pattern: {EXCEL_PATTERN})"
        )
    logger.info(f"Gefundene Dateien: {[f.name for f in excel_files]}")

    dfs = []
    for file in excel_files:
        df = pd.read_excel(file)
        df.columns = [str(c).lower().strip() for c in df.columns]
        df["_source_file"] = file.name
        dfs.append(df)
        logger.info(f"  {file.name}: {len(df)} Zeilen")

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Gesamt: {len(df)} Zeilen aus {len(excel_files)} Dateien")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Datum/Zeit parsen, Features extrahieren."""
    required = [COL_DATE, COL_TIME, COL_ISSUE, COL_PLACE]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Fehlende Spalten: {missing}. Vorhanden: {list(df.columns)}"
        )

    # Leere Issue/Date-Zeilen raus
    n_before = len(df)
    df = df.dropna(subset=[COL_DATE, COL_ISSUE]).copy()
    logger.info(f"Entfernte Zeilen ohne Datum/Issue: {n_before - len(df)}")

    # Place: leere → 'UNBEKANNT'
    df[COL_PLACE] = df[COL_PLACE].fillna("UNBEKANNT").astype(str).str.strip()
    df.loc[df[COL_PLACE] == "", COL_PLACE] = "UNBEKANNT"

    # Datum parsen (Format TT.MM.JJJJ)
    df["date_parsed"] = pd.to_datetime(
        df[COL_DATE].astype(str), dayfirst=True, errors="coerce"
    )
    n_before = len(df)
    df = df.dropna(subset=["date_parsed"]).copy()
    if n_before - len(df) > 0:
        logger.info(f"Entfernte Zeilen mit ungültigem Datum: {n_before - len(df)}")

    # ══════════════════════════════════════════════════════
    # ZEIT-PARSING (ohne Datum-Verschiebung!)
    # Das Datum in der Excel ist bereits korrekt.
    # ══════════════════════════════════════════════════════
    logger.info("Parse Zeitangaben (inkl. umgangssprachlicher)...")
    hours, confidences = parse_series(df[COL_TIME].tolist())

    df["hour"] = hours
    df["time_confidence"] = confidences  # 0=exakt, 1=fuzzy, 2=unknown

    # Statistik
    conf_counts = pd.Series(confidences).value_counts().to_dict()
    total = len(confidences)
    logger.info("Zeit-Parsing Statistik:")
    logger.info(f"  exakt   (16:00, 22 Uhr):      {conf_counts.get(0, 0):6d}  ({100*conf_counts.get(0, 0)/total:5.1f}%)")
    logger.info(f"  fuzzy   (morgen, abend, ...): {conf_counts.get(1, 0):6d}  ({100*conf_counts.get(1, 0)/total:5.1f}%)")
    logger.info(f"  unknown (Default 12:00):      {conf_counts.get(2, 0):6d}  ({100*conf_counts.get(2, 0)/total:5.1f}%)")

    # Datetime aus (korrektem) Datum + (geschätzter) Stunde
    df["datetime"] = df["date_parsed"] + pd.to_timedelta(df["hour"], unit="h")

    # Zeit-Features
    df["weekday"] = df["datetime"].dt.weekday
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["quarter"] = df["datetime"].dt.quarter
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    # Issue-Text säubern
    df[COL_ISSUE] = df[COL_ISSUE].astype(str).str.strip()

    df = df.sort_values("datetime").reset_index(drop=True)

    logger.info(f"Nach Preprocessing: {len(df)} Zeilen")
    logger.info(f"Zeitraum: {df['datetime'].min()} bis {df['datetime'].max()}")
    logger.info(f"Anzahl Bezirke (place): {df[COL_PLACE].nunique()}")

    return df


def load_and_preprocess() -> pd.DataFrame:
    df = load_all_excels()
    df = preprocess(df)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    df = load_and_preprocess()
    print("\n=== Vorschau (erste 10 Zeilen) ===")
    print(df[["datetime", "hour", "time_confidence", "weekday", "place", "issue"]].head(10))
    print(f"\nGesamt: {len(df)} Einträge")
    print(f"Bezirke: {df['place'].nunique()}")
    print(f"\nTop 10 Bezirke:")
    print(df['place'].value_counts().head(10))
