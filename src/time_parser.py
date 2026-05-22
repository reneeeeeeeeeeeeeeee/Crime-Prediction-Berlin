"""
Deutscher Zeit-Parser (angepasst)
─────────────────────────────────
Parst die 'time'-Spalte, die entweder eine exakte Uhrzeit (16:00)
oder eine umgangssprachliche Angabe (gestern nachmittag, heute morgen) enthält.

WICHTIG: Das Datum in der 'date'-Spalte ist bereits korrekt auf das
Ereignis-Datum angepasst (nicht auf das Melde-Datum). Wenn "gestern" im
time-Feld steht, wurde die date-Spalte bereits einen Tag zurückgesetzt.
Daher verschieben wir das Datum NICHT mehr – wir extrahieren nur noch
die ungefähre Stunde aus dem Tageszeit-Ausdruck.

Rückgabe:
  - hour: geschätzte Stunde (0-23)
  - confidence: 0=exakt, 1=fuzzy, 2=unknown
"""
import re
from typing import Tuple, Union

# ──────────────────────────────────────────────
# Tageszeit-Ausdrücke → Stunde
# ──────────────────────────────────────────────
TIME_EXPRESSIONS = {
    "nachts":           2,
    "nacht":            2,
    "früh morgens":     6,
    "frühmorgens":      6,
    "morgens":          8,
    "morgen früh":      7,
    "morgen":           8,
    "vormittags":      10,
    "vormittag":       10,
    "mittags":         12,
    "mittag":          12,
    "mittagszeit":     12,
    "nachmittags":     15,
    "nachmittag":      15,
    "später nachmittag": 17,
    "abends":          20,
    "abend":           20,
    "früher abend":    18,
    "später abend":    22,
    "spät abends":     22,
    "spätabends":      22,
}


def _normalize(text: str) -> str:
    s = str(text).lower().strip()
    s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    s = re.sub(r"\s+", " ", s)
    return s


_TIME_NORM = {_normalize(k): v for k, v in TIME_EXPRESSIONS.items()}


def parse_time_field(time_value: Union[str, float, None]) -> Tuple[int, int]:
    """
    Parst einen Eintrag der 'time'-Spalte.

    Returns:
        (hour, confidence)
        confidence: 0=exakt, 1=fuzzy, 2=unknown (Default 12:00)
    """
    if time_value is None or (isinstance(time_value, float) and str(time_value) == "nan"):
        return 12, 2

    s = str(time_value).strip()
    if s == "" or s.lower() in ("nan", "nat", "none"):
        return 12, 2

    # 1. Exakte Zeit (HH:MM / HH:MM:SS)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", s)
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h, 0

    # 2. Nur Stunde ("16" oder "16 Uhr")
    m = re.match(r"^(\d{1,2})\s*(uhr)?$", s.lower())
    if m:
        h = int(m.group(1))
        if 0 <= h <= 23:
            return h, 0

    # 3. Tageszeit-Ausdruck (längste Keywords zuerst)
    normalized = _normalize(s)
    for keyword in sorted(_TIME_NORM.keys(), key=len, reverse=True):
        if keyword in normalized:
            return _TIME_NORM[keyword], 1  # fuzzy

    # 4. Default
    return 12, 2


def parse_series(time_series):
    """Batch-Version. Gibt zwei Listen zurück: hours, confidences."""
    hours = []
    confidences = []
    for t in time_series:
        h, c = parse_time_field(t)
        hours.append(h)
        confidences.append(c)
    return hours, confidences


if __name__ == "__main__":
    test_cases = [
        ("16:00",              "Exakte Zeit"),
        ("08:30",              "Exakte Zeit"),
        ("22",                 "Nur Stunde"),
        ("16 Uhr",             "Mit 'Uhr'"),
        ("morgen",             "Tageszeit → 08"),
        ("abend",              "Tageszeit → 20"),
        ("nachmittag",         "Tageszeit → 15"),
        ("mittag",             "Tageszeit → 12"),
        ("nacht",              "Tageszeit → 02"),
        ("gestern nachmittag", "Fuzzy, Stunde aus 'nachmittag'"),
        ("gestern abend",      "Fuzzy, Stunde aus 'abend'"),
        ("letzte nacht",       "Fuzzy, Stunde aus 'nacht'"),
        ("heute morgen",       "Fuzzy, Stunde aus 'morgen'"),
        ("heute",              "Nur Datum-Wort → unknown"),
        ("gestern",            "Nur Datum-Wort → unknown"),
        ("spätabends",         "Fuzzy → 22"),
        ("irgendwas",          "Unklar → Default 12"),
        ("",                   "Leer"),
        (None,                 "None"),
    ]

    conf_names = {0: "EXAKT  ", 1: "FUZZY  ", 2: "UNKNOWN"}
    print(f"{'Input':<25} {'Erwartung':<40} {'Stunde':<8} {'Confidence'}")
    print("-" * 90)
    for value, label in test_cases:
        h, c = parse_time_field(value)
        print(f"{str(value):<25} {label:<40} {h:02d}:00    {conf_names[c]}")
