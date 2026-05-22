"""
SCHRITT 5.1: Streamlit App – Tagesübersicht + Bezirk × Datum × Zeitfenster
───────────────────────────────────────────────────────────────────────────
Neu in v5.1:
  - Tab 2: Gezielter Blick auf einen Bezirk mit Datum + Zeitfenster
  - Zeit-Match-Score: Wie stark überschneidet sich das Zeitfenster mit den
    historischen Peaks je Kategorie?
  - Sortierung nach Zeit-Match × Erwartungswert
"""
import json
import math
import pickle
from pathlib import Path
from datetime import datetime, timedelta, date, time as dtime

import numpy as np
import pandas as pd
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import CACHE_DIR, MODELS_DIR, COL_ISSUE, COL_PLACE

# ──────────────────────────────────────────────
# STREAMLIT CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Event Prediction", page_icon="🔮", layout="wide")

st.markdown("""
<style>
    .main, .stApp { background-color: #0f1117; }
    .top-card {
        background: linear-gradient(135deg, #1a1d2e, #252840);
        border: 1px solid #3a3f6e;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .top-card.match-high {
        border-color: #4a8f5e;
        background: linear-gradient(135deg, #1a2e22, #1f3328);
    }
    .top-card.match-mid {
        border-color: #7a7a3e;
        background: linear-gradient(135deg, #28271a, #2e2d1f);
    }
    .top-card.match-none {
        opacity: 0.45;
    }
    .rank-badge {
        font-size: 22px; font-weight: 800; color: #7c83ff;
        min-width: 42px; display: inline-block;
    }
    .count-badge {
        font-size: 24px; font-weight: 700; color: #a0ffb0;
        margin-right: 8px;
    }
    .prob-badge {
        font-size: 16px; color: #ffb0a0; font-weight: 600;
        margin-left: 8px;
    }
    .match-badge-high {
        font-size: 13px; color: #70ffaa; font-weight: 700;
        background: #1a4030; border-radius: 10px;
        padding: 2px 10px; margin-left: 8px;
    }
    .match-badge-mid {
        font-size: 13px; color: #ffe080; font-weight: 700;
        background: #3a3010; border-radius: 10px;
        padding: 2px 10px; margin-left: 8px;
    }
    .match-badge-none {
        font-size: 13px; color: #888; font-weight: 400;
        background: #1a1a1a; border-radius: 10px;
        padding: 2px 10px; margin-left: 8px;
    }
    .count-unit { font-size: 13px; color: #888; font-weight: 400; }
    .tag {
        display: inline-block; background: #2a2f52; color: #a0a8ff;
        border-radius: 20px; padding: 3px 12px; font-size: 12px;
        margin-right: 6px; margin-top: 4px;
    }
    .tag-cat { background: #3d2a4d; color: #d0a0ff; }
    .tag-place { background: #2a4d52; color: #a0ffd0; }
    .example-text {
        color: #888; font-size: 12px; font-style: italic;
        margin-top: 8px; border-left: 2px solid #3a3f6e; padding-left: 10px;
    }
    .time-hint {
        color: #ffd0a0; font-size: 11px; margin-top: 4px;
        margin-left: 50px;
    }
    .summary-box {
        background: #1a1d2e; border-radius: 12px;
        padding: 16px; margin-bottom: 20px;
        border: 1px solid #3a3f6e;
    }
    .summary-box-green {
        background: #111e16; border-radius: 12px;
        padding: 16px; margin-bottom: 20px;
        border: 1px solid #2a6040;
    }
    h1, h2, h3 { color: #ffffff !important; }
    .section-label {
        font-size: 11px; color: #666; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 4px;
    }
    .zeit-fenster-info {
        background: #1a2030; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 12px;
        border-left: 3px solid #4a7aff; font-size: 13px; color: #aac4ff;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    reg_path = MODELS_DIR / "count_regressors.pkl"
    enc_path = MODELS_DIR / "count_place_encoder.pkl"
    feat_path = MODELS_DIR / "count_feature_names.json"
    meta_path = MODELS_DIR / "count_metadata.json"

    if not reg_path.exists():
        raise FileNotFoundError("Count-Modelle fehlen. Bitte 3c_train_counts.py ausführen.")

    with open(reg_path, "rb") as f:
        models = pickle.load(f)
    with open(enc_path, "rb") as f:
        place_enc = pickle.load(f)
    with open(feat_path) as f:
        feature_names = json.load(f)
    with open(meta_path) as f:
        metadata = json.load(f)

    df = pd.read_csv(CACHE_DIR / "classified_data.csv", encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])

    return {
        "models": models,
        "place_enc": place_enc,
        "feature_names": feature_names,
        "metadata": metadata,
        "df": df,
    }


# ──────────────────────────────────────────────
# ZEITFENSTER-BERECHNUNG
# ──────────────────────────────────────────────
def find_tight_time_window(hours: np.ndarray, coverage: float = 0.6) -> tuple:
    if len(hours) < 2:
        h = int(hours[0]) if len(hours) > 0 else 12
        return (h, h, 100.0)

    counts = np.bincount(hours.astype(int), minlength=24)
    total = counts.sum()
    need = total * coverage

    best_window = None
    best_span = 25

    for start in range(24):
        cumulative = 0
        for span in range(1, 25):
            hour = (start + span - 1) % 24
            cumulative += counts[hour]
            if cumulative >= need:
                if span < best_span:
                    best_span = span
                    end = (start + span - 1) % 24
                    coverage_actual = cumulative / total * 100
                    best_window = (start, end, coverage_actual, span)
                break

    if best_window is None:
        top = counts.argmax()
        return (int(top), int(top), 100.0)

    return best_window[0], best_window[1], best_window[2]


@st.cache_data
def build_time_profiles(_df: pd.DataFrame) -> dict:
    df = _df.copy()
    if "time_confidence" in df.columns:
        df = df[df["time_confidence"] == 0]

    profiles = {}

    for (place, cat), group in df.groupby([COL_PLACE, "category"]):
        if len(group) < 3:
            continue
        hours = group["hour"].values
        start, end, cov = find_tight_time_window(hours, coverage=0.6)
        top_hour = int(group["hour"].value_counts().index[0])
        profiles[(place, cat)] = {
            "start": start,
            "end": end,
            "top_hour": top_hour,
            "coverage": cov,
            "n_samples": len(group),
            "hour_counts": np.bincount(hours.astype(int), minlength=24).tolist(),
        }

    for cat, group in df.groupby("category"):
        key = ("__ALL__", cat)
        if len(group) < 3:
            continue
        hours = group["hour"].values
        start, end, cov = find_tight_time_window(hours, coverage=0.6)
        top_hour = int(group["hour"].value_counts().index[0])
        profiles[key] = {
            "start": start,
            "end": end,
            "top_hour": top_hour,
            "coverage": cov,
            "n_samples": len(group),
            "hour_counts": np.bincount(hours.astype(int), minlength=24).tolist(),
        }
    return profiles


def circular_hours(start: int, end: int) -> set:
    """Gibt die Menge aller Stunden im Fenster [start, end] zurück (kreisförmig)."""
    if end >= start:
        return set(range(start, end + 1))
    else:
        return set(range(start, 24)) | set(range(0, end + 1))


def time_match_score(q_start: int, q_end: int, profile: dict) -> float:
    """
    Berechnet wie gut das Abfrage-Zeitfenster zum historischen Profil passt.
    Gibt einen Score 0.0–1.0 zurück, basierend auf dem Anteil der Profilstunden
    im Abfragefenster, gewichtet nach tatsächlicher Häufigkeit.

    1.0 = Abfrage-Fenster trifft den Peak vollständig
    0.0 = keine Überschneidung
    """
    if not profile:
        return 0.0

    hour_counts = np.array(profile.get("hour_counts", [0] * 24), dtype=float)
    total = hour_counts.sum()
    if total == 0:
        return 0.0

    query_hours = circular_hours(q_start, q_end)
    matched = sum(hour_counts[h] for h in query_hours if h < 24)
    return float(matched / total)


def format_time_hint(profile: dict) -> str:
    if not profile:
        return ""
    start = profile["start"]
    end = profile["end"]
    cov = profile["coverage"]
    n = profile["n_samples"]

    if end >= start:
        span = end - start + 1
    else:
        span = (24 - start) + end + 1

    if start == end:
        return f"🕐 meist um {start:02d}:00 Uhr ({cov:.0f}% der Fälle, n={n})"
    if span <= 4:
        return f"🕐 häufig {start:02d}–{end:02d} Uhr ({cov:.0f}% der Fälle, n={n})"
    if span <= 8:
        top_h = profile["top_hour"]
        return f"🕐 meist {start:02d}–{end:02d} Uhr, häufigste Stunde {top_h:02d}:00 (n={n})"
    return _describe_wide_window(start, end, profile["top_hour"], n)


def _describe_wide_window(start: int, end: int, top: int, n: int) -> str:
    if 6 <= top <= 11:
        zeit = "vormittags/mittags"
    elif 11 <= top <= 14:
        zeit = "mittags"
    elif 14 <= top <= 18:
        zeit = "nachmittags"
    elif 18 <= top <= 22:
        zeit = "abends"
    elif 22 <= top <= 24 or 0 <= top <= 2:
        zeit = "nachts"
    elif 2 <= top <= 6:
        zeit = "frühmorgens"
    else:
        zeit = "tagsüber"
    return f"🕐 verteilt, Schwerpunkt {zeit} (häufigste Stunde {top:02d}:00, n={n})"


def format_match_badge(score: float) -> tuple[str, str]:
    """Gibt (HTML-Badge, CSS-Klasse) zurück."""
    if score >= 0.35:
        pct = int(score * 100)
        return f'<span class="match-badge-high">⏱ {pct}% Match</span>', "match-high"
    elif score >= 0.1:
        pct = int(score * 100)
        return f'<span class="match-badge-mid">⏱ {pct}% Match</span>', "match-mid"
    else:
        return '<span class="match-badge-none">außerhalb Zeitfenster</span>', "match-none"


# ──────────────────────────────────────────────
# WAHRSCHEINLICHKEIT
# ──────────────────────────────────────────────
def lambda_to_probability(lam: float) -> float:
    return 1.0 - math.exp(-lam)


# ──────────────────────────────────────────────
# FEATURE BUILDER & PREDICTION
# ──────────────────────────────────────────────
def build_feature_row(target_date: date, place_code: int, data_start: date) -> np.ndarray:
    wd = target_date.weekday()
    mo = target_date.month
    dy = target_date.day
    doy = target_date.timetuple().tm_yday
    is_weekend = int(wd in [5, 6])
    year = target_date.year
    days_since_start = (target_date - data_start).days

    wd_sin = np.sin(2 * np.pi * wd / 7)
    wd_cos = np.cos(2 * np.pi * wd / 7)
    mo_sin = np.sin(2 * np.pi * mo / 12)
    mo_cos = np.cos(2 * np.pi * mo / 12)
    doy_sin = np.sin(2 * np.pi * doy / 365)
    doy_cos = np.cos(2 * np.pi * doy / 365)

    return np.array([
        wd_sin, wd_cos, mo_sin, mo_cos, doy_sin, doy_cos,
        wd, mo, dy, is_weekend, year, days_since_start,
        place_code,
    ], dtype=np.float64)


def predict_day(target_date: date, place_filter: str, artifacts: dict) -> pd.DataFrame:
    models = artifacts["models"]
    place_enc = artifacts["place_enc"]
    data_start = datetime.fromisoformat(artifacts["metadata"]["data_start_date"]).date()

    places = list(place_enc.classes_) if place_filter == "ALL" else [place_filter]

    rows = []
    for place_name in places:
        place_code = int(place_enc.transform([place_name])[0])
        feat = build_feature_row(target_date, place_code, data_start).reshape(1, -1)
        for cat_name, model in models.items():
            lam = float(np.clip(model.predict(feat)[0], 0, None))
            rows.append({
                "Bezirk": place_name,
                "Kategorie": cat_name,
                "Erwartet": lam,
                "Wahrscheinlichkeit": lambda_to_probability(lam),
            })

    return pd.DataFrame(rows).sort_values("Erwartet", ascending=False).reset_index(drop=True)


def find_example(df: pd.DataFrame, bezirk: str, kategorie: str) -> str:
    subset = df[(df[COL_PLACE] == bezirk) & (df["category"] == kategorie)]
    if len(subset) == 0:
        subset = df[df["category"] == kategorie]
    if len(subset) == 0:
        return ""
    return str(subset.sort_values("datetime", ascending=False).iloc[0][COL_ISSUE])[:180]


def lookup_time_profile(profiles: dict, place: str, cat: str) -> dict:
    if (place, cat) in profiles:
        return profiles[(place, cat)]
    if ("__ALL__", cat) in profiles:
        return profiles[("__ALL__", cat)]
    return {}


# ──────────────────────────────────────────────
# RENDER: Tagesübersicht-Karten (bestehend)
# ──────────────────────────────────────────────
def render_prediction_cards(rows: pd.DataFrame, df_source: pd.DataFrame, profiles: dict):
    if len(rows) == 0:
        st.warning("Keine Vorhersagen verfügbar.")
        return

    detail = rows.copy()
    detail.index = range(1, len(detail) + 1)

    for rank, row in detail.iterrows():
        example = find_example(df_source, row["Bezirk"], row["Kategorie"])
        prof = lookup_time_profile(profiles, row["Bezirk"], row["Kategorie"])
        time_hint = format_time_hint(prof)
        prob_percent = row["Wahrscheinlichkeit"] * 100

        st.markdown(f"""
        <div class="top-card">
            <span class="rank-badge">#{rank}</span>
            <span class="count-badge">{row['Erwartet']:.2f}</span>
            <span class="count-unit">erwartet</span>
            <span class="prob-badge">~{prob_percent:.0f}% Wahrscheinlichkeit</span>
            &nbsp;
            <span class="tag tag-place">📍 {row['Bezirk']}</span>
            <span class="tag tag-cat">📂 {row['Kategorie']}</span>
            <div class="time-hint">{time_hint}</div>
            <div class="example-text">📌 {example}</div>
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# RENDER: Bezirk × Zeit Karten (neu)
# ──────────────────────────────────────────────
def render_bezirk_zeit_cards(
    rows: pd.DataFrame,
    df_source: pd.DataFrame,
    profiles: dict,
    q_start: int,
    q_end: int,
    hide_no_match: bool,
    query_date: date = None,
):
    if len(rows) == 0:
        st.warning("Keine Vorhersagen verfügbar.")
        return

    shown = 0
    for rank_raw, (_, row) in enumerate(rows.iterrows(), 1):
        prof = lookup_time_profile(profiles, row["Bezirk"], row["Kategorie"])
        score = time_match_score(q_start, q_end, prof)
        badge_html, card_class = format_match_badge(score)

        if hide_no_match and score < 0.05:
            continue

        example = find_example(df_source, row["Bezirk"], row["Kategorie"])
        time_hint = format_time_hint(prof)
        shown += 1

        pct = int(score * 100)
        date_tag = f'<span class="tag">📅 {query_date.strftime("%d.%m.%Y")}</span>' if query_date else ""
        st.markdown(f"""
        <div class="top-card {card_class}">
            <span class="rank-badge">#{shown}</span>
            <span class="count-badge">{pct}%</span>
            <span class="count-unit">Zeitfenster-Match</span>
            &nbsp;
            {date_tag}
            <span class="tag tag-place">📍 {row['Bezirk']}</span>
            <span class="tag tag-cat">📂 {row['Kategorie']}</span>
            <div class="time-hint">{time_hint}</div>
            <div class="example-text">📌 {example}</div>
        </div>
        """, unsafe_allow_html=True)

    if shown == 0:
        st.info("Keine Kategorien mit Zeitfenster-Überschneidung gefunden. "
                "Deaktiviere 'Nur passende Zeiten', um alle anzuzeigen.")


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("# 🔮 Event Prediction")

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Fehler: {e}")
    st.info("Bitte zuerst ausführen: `python src/3c_train_counts.py`")
    st.stop()

time_profiles = build_time_profiles(artifacts["df"])
all_bezirke = sorted(list(artifacts["place_enc"].classes_))

# ──────────────────────────────────────────────
# SIDEBAR: Gemeinsame Infos
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Daten")
    df_info = artifacts["df"]
    st.metric("Trainingsmeldungen", f"{len(df_info):,}")
    st.metric("Kategorien", len(artifacts["models"]))
    st.metric("Bezirke", len(artifacts["place_enc"].classes_))

    md = artifacts["metadata"]
    st.markdown("---")
    st.markdown("### 🕒 Modell-Info")
    st.caption(f"Daten: {md['data_start_date']} → {md['data_end_date']}")
    st.caption("Aktuellere Daten zählen mehr")

# ──────────────────────────────────────────────
# ZWEI HAUPT-TABS
# ──────────────────────────────────────────────
tab1, tab2 = st.tabs(["📅  Tagesübersicht", "🔍  Bezirk & Zeitfenster"])


# ══════════════════════════════════════════════
# TAB 1: Tagesübersicht (bestehend)
# ══════════════════════════════════════════════
with tab1:
    st.markdown("#### Erwartete Meldungen pro Tag mit Uhrzeit-Kontext & Wahrscheinlichkeit")

    col_date, col_space = st.columns([2, 5])
    with col_date:
        target = st.date_input(
            "Datum:",
            value=date.today() + timedelta(days=1),
            key="tab1_date",
        )

    weekday_names = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                     "Freitag", "Samstag", "Sonntag"]
    st.markdown(f"### {target.strftime('%d.%m.%Y')} – {weekday_names[target.weekday()]}")
    st.caption("Ranking über alle Bezirke und Kategorien. Uhrzeit aus historischen Profilen.")
    st.markdown("---")

    with st.spinner("Berechne erwartete Mengen..."):
        full = predict_day(target, "ALL", artifacts)

    total_events = full["Erwartet"].sum()
    cat_totals = full.groupby("Kategorie", as_index=False).agg(
        Erwartet=("Erwartet", "sum")
    ).sort_values("Erwartet", ascending=False)
    place_totals = full.groupby("Bezirk", as_index=False)["Erwartet"].sum().sort_values(
        "Erwartet", ascending=False
    )
    top_place = place_totals.iloc[0]
    top_cat = cat_totals.iloc[0]

    st.markdown(f"""
    <div class="summary-box">
        <h3 style="margin:0;">📈 Tagesprognose</h3>
        <p style="font-size:32px; font-weight:700; color:#a0ffb0; margin:8px 0;">
            {total_events:.0f} <span style="font-size:16px; color:#888;">Meldungen erwartet</span>
        </p>
        <p style="color:#aaa; margin:0;">
            Stärkster Bezirk: {top_place['Bezirk']} ({top_place['Erwartet']:.1f}) ·
            Stärkste Kategorie: {top_cat['Kategorie']} ({top_cat['Erwartet']:.1f})
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Top-Kategorien des Tages")
    shown = 0
    for _, row in cat_totals.iterrows():
        if row["Erwartet"] < 0.1:
            continue
        if shown >= 10:
            break
        shown += 1
        prob_percent = lambda_to_probability(row["Erwartet"]) * 100
        prof = lookup_time_profile(time_profiles, "__ALL__", row["Kategorie"])
        time_hint = format_time_hint(prof)
        st.markdown(f"""
        <div class="top-card">
            <span class="rank-badge">#{shown}</span>
            <span class="count-badge">{row['Erwartet']:.1f}</span>
            <span class="count-unit">erwartet</span>
            <span class="prob-badge">~{prob_percent:.0f}% Wahrscheinlichkeit</span>
            &nbsp;
            <span class="tag tag-cat">📂 {row['Kategorie']}</span>
            <div class="time-hint">{time_hint}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📋 Was passiert voraussichtlich wann und wo?")
    t30, t50 = st.tabs(["Top 30", "Top 50"])
    with t30:
        render_prediction_cards(full.head(30), artifacts["df"], time_profiles)
    with t50:
        render_prediction_cards(full.head(50), artifacts["df"], time_profiles)

    st.markdown("### 🗺️ Verteilung pro Bezirk")
    place_df = place_totals.head(20).copy()
    place_df["Erwartet"] = place_df["Erwartet"].round(1)
    st.bar_chart(place_df.set_index("Bezirk")["Erwartet"])


# ══════════════════════════════════════════════
# TAB 2: Bezirk × Zeitfenster (neu in v5.1)
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### Welche Vorfälle sind in einem Bezirk zu einer bestimmten Zeit zu erwarten?")

    # ── Eingabe-Zeile ──
    col_bez, col_dat, col_von, col_bis, col_opt = st.columns([2, 2, 1.5, 1.5, 2])

    with col_bez:
        st.markdown('<div class="section-label">Bezirk</div>', unsafe_allow_html=True)
        selected_bezirk = st.selectbox(
            "Bezirk",
            options=all_bezirke,
            label_visibility="collapsed",
            key="t2_bezirk",
        )

    with col_dat:
        st.markdown('<div class="section-label">Datum</div>', unsafe_allow_html=True)
        target2 = st.date_input(
            "Datum",
            value=date.today() + timedelta(days=1),
            label_visibility="collapsed",
            key="t2_date",
        )

    with col_von:
        st.markdown('<div class="section-label">Von (Uhr)</div>', unsafe_allow_html=True)
        time_von = st.time_input(
            "Von",
            value=dtime(8, 0),
            step=3600,
            label_visibility="collapsed",
            key="t2_von",
        )

    with col_bis:
        st.markdown('<div class="section-label">Bis (Uhr)</div>', unsafe_allow_html=True)
        time_bis = st.time_input(
            "Bis",
            value=dtime(18, 0),
            step=3600,
            label_visibility="collapsed",
            key="t2_bis",
        )

    with col_opt:
        st.markdown('<div class="section-label">Optionen</div>', unsafe_allow_html=True)
        hide_no_match = st.toggle(
            "Nur passende Zeiten",
            value=True,
            key="t2_hide",
            help="Kategorien ausblenden, die außerhalb des Zeitfensters liegen",
        )
        sort_by_match = st.toggle(
            "Nach Zeit-Match sortieren",
            value=True,
            key="t2_sort",
            help="Zuerst Kategorien die am besten zum Zeitfenster passen",
        )

    q_start = time_von.hour
    q_end = time_bis.hour

    # Mitternacht-Überschreitung anzeigen
    if q_end < q_start:
        crossing = f"(über Mitternacht: {q_start:02d}:00 → {q_end:02d}:00)"
    else:
        crossing = ""

    weekday_names2 = ["Montag", "Dienstag", "Mittwoch", "Donnerstag",
                      "Freitag", "Samstag", "Sonntag"]
    st.markdown(f"""
    <div class="zeit-fenster-info">
        📍 <b>{selected_bezirk}</b> &nbsp;·&nbsp;
        📅 <b>{target2.strftime('%d.%m.%Y')}</b> ({weekday_names2[target2.weekday()]}) &nbsp;·&nbsp;
        🕐 <b>{q_start:02d}:00 – {q_end:02d}:00 Uhr</b> {crossing}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Vorhersage laden ──
    with st.spinner(f"Berechne Vorhersage für {selected_bezirk}..."):
        preds = predict_day(target2, selected_bezirk, artifacts)

    # ── Zeit-Match-Score hinzufügen ──
    match_scores = []
    for _, row in preds.iterrows():
        prof = lookup_time_profile(time_profiles, row["Bezirk"], row["Kategorie"])
        match_scores.append(time_match_score(q_start, q_end, prof))
    preds["ZeitMatch"] = match_scores

    # ── Sortierung ──
    if sort_by_match:
        preds["SortScore"] = preds["ZeitMatch"] * preds["Erwartet"]
        preds = preds.sort_values("SortScore", ascending=False).drop(columns=["SortScore"])
    # else: bleibt nach Erwartet sortiert (Standard aus predict_day)

    # ── Summary ──
    matching = preds[preds["ZeitMatch"] >= 0.1]
    total_match = matching["Erwartet"].sum()
    total_all = preds["Erwartet"].sum()

    st.markdown(f"""
    <div class="summary-box-green">
        <h3 style="margin:0;">📊 Bezirks-Prognose: {selected_bezirk}</h3>
        <p style="font-size:28px; font-weight:700; color:#a0ffb0; margin:8px 0;">
            {total_all:.0f}
            <span style="font-size:15px; color:#888;">Meldungen erwartet (ganzer Tag)</span>
        </p>
        <p style="color:#aaa; margin:0;">
            Im Zeitfenster {q_start:02d}:00–{q_end:02d}:00 aktiv:
            <b style="color:#70ffaa;">{len(matching)} Kategorien</b>
            mit zusammen <b style="color:#70ffaa;">{total_match:.1f}</b> erwarteten Meldungen
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Karten ──
    render_bezirk_zeit_cards(
        preds,
        artifacts["df"],
        time_profiles,
        q_start,
        q_end,
        hide_no_match,
        query_date=target2,
    )
