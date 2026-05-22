import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent))

from topk_forecasting import add_time_bucket
from topk_forecasting_v2 import load_topk_v2_artifacts, predict_v2_rankings
from topk_forecasting_v3 import filter_standard_places, load_topk_v3_artifacts, predict_v3_rankings


BEZIRK_COORDS = {
    "Mitte":                        (52.5200, 13.3900),
    "Friedrichshain-Kreuzberg":     (52.5000, 13.4500),
    "Pankow":                       (52.5667, 13.4167),
    "Charlottenburg-Wilmersdorf":   (52.5050, 13.2950),
    "Spandau":                      (52.5353, 13.2000),
    "Steglitz-Zehlendorf":          (52.4333, 13.3200),
    "Tempelhof-Schöneberg":         (52.4667, 13.3833),
    "Neukölln":                     (52.4667, 13.4333),
    "Treptow-Köpenick":             (52.4200, 13.5800),
    "Marzahn-Hellersdorf":          (52.5333, 13.5833),
    "Lichtenberg":                  (52.5167, 13.5000),
    "Reinickendorf":                (52.5833, 13.3333),
}


st.set_page_config(page_title="Event Prediction Top-K", layout="wide")

st.markdown(
    """
<style>
    .main, .stApp { background-color: #0f1117; }
    .metric-card {
        background: linear-gradient(135deg, #171a28, #22263a);
        border: 1px solid #3a3f6e;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .pred-card {
        background: linear-gradient(135deg, #171a28, #242840);
        border: 1px solid #3a3f6e;
        border-radius: 14px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .rank {
        font-size: 22px;
        font-weight: 800;
        color: #7c83ff;
        min-width: 46px;
        display: inline-block;
    }
    .prob {
        font-size: 22px;
        font-weight: 800;
        color: #a0ffb0;
        margin-right: 10px;
    }
    .lift {
        font-size: 16px;
        font-weight: 700;
        color: #ffd27f;
        margin-left: 10px;
    }
    .subtle {
        color: #9ba3c7;
        font-size: 13px;
    }
    .tag {
        display: inline-block;
        border-radius: 999px;
        padding: 4px 12px;
        font-size: 12px;
        margin-right: 6px;
        margin-top: 6px;
    }
    .tag-bucket { background: #4a3421; color: #ffd4a3; }
    .tag-place { background: #214447; color: #b2fff3; }
    .tag-cat { background: #41264b; color: #f0b3ff; }
    .example {
        color: #b0b6cf;
        font-size: 12px;
        margin-top: 8px;
        border-left: 2px solid #3a3f6e;
        padding-left: 10px;
    }
    h1, h2, h3, h4 { color: #ffffff !important; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_app_artifacts():
    last_exc = None
    for version, loader in [
        ("V3", load_topk_v3_artifacts),
        ("V2", load_topk_v2_artifacts),
    ]:
        try:
            artifacts = loader(include_data=True)
            exact = filter_standard_places(artifacts["df"])
            exact = exact[exact["time_confidence"] == 0].copy()
            exact = add_time_bucket(exact)
            artifacts["exact_df"] = exact
            artifacts["model_version"] = version
            return artifacts
        except Exception as exc:
            last_exc = exc
    raise last_exc


def find_example(exact_df: pd.DataFrame, place: str, category: str, bucket: str) -> str:
    subset = exact_df[
        (exact_df["place"] == place)
        & (exact_df["category"] == category)
        & (exact_df["time_bucket"] == bucket)
    ]
    if len(subset) == 0:
        subset = exact_df[
            (exact_df["place"] == place)
            & (exact_df["category"] == category)
        ]
    if len(subset) == 0:
        subset = exact_df[exact_df["category"] == category]
    if len(subset) == 0:
        return ""
    return str(subset.sort_values("datetime", ascending=False).iloc[0]["issue"])[:180]


def render_rows(rows: pd.DataFrame, exact_df: pd.DataFrame, mode: str):
    if len(rows) == 0:
        st.warning("Keine Vorhersagen verfuegbar.")
        return

    detail = rows.copy()
    detail.index = range(1, len(detail) + 1)

    for rank, row in detail.iterrows():
        example = find_example(exact_df, row["Bezirk"], row["Kategorie"], row["Zeitfenster"])
        if mode == "lift":
            trailing = f"x{row['Lift']:.2f} Lift"
        else:
            trailing = f"x{row['Lift']:.2f} gg. Basis"

        st.markdown(
            f"""
        <div class="pred-card">
            <span class="rank">#{rank}</span>
            <span class="prob">{row['Wahrscheinlichkeit'] * 100:.1f}%</span>
            <span class="subtle">Wahrscheinlichkeit in der Zelle</span>
            <span class="lift">{trailing}</span>
            <div style="margin-top:8px;">
                <span class="tag tag-bucket">{row['Zeitfenster']} Uhr</span>
                <span class="tag tag-place">{row['Bezirk']}</span>
                <span class="tag tag-cat">{row['Kategorie']}</span>
            </div>
            <div class="subtle" style="margin-top:8px;">
                Erwartet: {row['Erwartet']:.3f} ·
                Zellvolumen: {row['Zellenwert']:.3f} ·
                Anteil: {row['Anteil'] * 100:.1f}% ·
                Basisanteil: {row['BasisAnteil'] * 100:.1f}%
            </div>
            <div class="example">{example}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


try:
    artifacts = load_app_artifacts()
except Exception as exc:
    st.error(f"Fehler: {exc}")
    st.info("Bitte zuerst ausfuehren: `python src/3f_train_topk_v3.py` oder `python src/3e_train_topk_v2.py`")
    st.stop()

model_version = artifacts["model_version"]
metadata = artifacts["metadata"]

st.markdown(f"# Event Prediction Top-K {model_version}")
st.markdown(f"#### Architektur: {metadata['architecture']}")
if model_version == "V3":
    st.caption("V3 nutzt rekursive Count-Forecasts pro Bezirk x Zeitfenster und ist fuer Zukunftstage ab dem naechsten Stichtag gedacht.")
else:
    st.caption("V2 nutzt die faktorisierte count_total/category_share-Architektur ohne rekursive Lags.")


with st.sidebar:
    st.markdown("## Eingabe")
    if model_version == "V3":
        min_date = pd.to_datetime(metadata["data_end_date"]).date() + timedelta(days=1)
        default_date = max(date.today() + timedelta(days=1), min_date)
        target_date = st.date_input("Datum", value=default_date, min_value=min_date)
        st.caption(f"V3 verfuegbar ab {min_date.strftime('%d.%m.%Y')}.")
    else:
        default_date = date.today() + timedelta(days=1)
        target_date = st.date_input("Datum", value=default_date)
    st.caption("Das Modell rankt absolute Wahrscheinlichkeit und relativen Lift getrennt.")

    st.markdown("---")
    st.markdown("### Diversity")
    max_per_cat = st.slider(
        "Max. Eintraege pro Kategorie",
        min_value=1,
        max_value=20,
        value=5,
        help="Begrenzt wie oft dieselbe Kategorie im Ranking auftauchen darf. 1 = maximale Vielfalt.",
    )

    st.markdown("---")
    st.markdown("### Modell")
    recall30 = artifacts.get("evaluation", {}).get("topk_backtest", {}).get("recall_at_30", 0.0)
    st.metric("Kategorien", metadata["n_valid_categories"])
    st.metric("Bezirke", metadata["n_places"])
    st.metric("Buckets", len(metadata["bucket_labels"]))
    st.caption(f"Daten: {metadata['data_start_date']} -> {metadata['data_end_date']}")
    st.caption(f"Exakte Uhrzeiten: {metadata.get('exact_time_share', 0.0) * 100:.1f}%")
    st.caption(f"Backtest Recall@30: {recall30:.3f}")


with st.spinner("Berechne Rankings ..."):
    predictor = predict_v3_rankings if model_version == "V3" else predict_v2_rankings
    try:
        raw_df = artifacts.get("df")
        predictions_absolute = predictor(target_date, artifacts, sort_by="absolute", max_per_category=max_per_cat, df=raw_df)
        predictions_lift = predictor(target_date, artifacts, sort_by="lift", max_per_category=max_per_cat, df=raw_df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

total_expected = predictions_absolute["Erwartet"].sum()
top_places = (
    predictions_absolute.groupby("Bezirk", as_index=False)["Erwartet"]
    .sum()
    .sort_values("Erwartet", ascending=False)
)
top_categories = (
    predictions_absolute.groupby("Kategorie", as_index=False)["Erwartet"]
    .sum()
    .sort_values("Erwartet", ascending=False)
)

top_place = top_places.iloc[0]
top_category = top_categories.iloc[0]
top_lift = predictions_lift.iloc[0]

st.markdown(
    f"""
<div class="metric-card">
    <h3 style="margin:0;">Prognose fuer {target_date.strftime('%d.%m.%Y')}</h3>
    <p style="font-size:34px; font-weight:800; color:#a0ffb0; margin:8px 0;">
        {total_expected:.1f} <span style="font-size:16px; color:#9ba3c7;">erwartete Meldungen mit Zeitfenster</span>
    </p>
    <p class="subtle" style="margin:0;">
        Staerkster Bezirk: {top_place['Bezirk']} ({top_place['Erwartet']:.2f}) ·
        Staerkste Kategorie: {top_category['Kategorie']} ({top_category['Erwartet']:.2f}) ·
        Hoechster Lift: {top_lift['Kategorie']} in {top_lift['Bezirk']} {top_lift['Zeitfenster']} (x{top_lift['Lift']:.2f})
    </p>
</div>
""",
    unsafe_allow_html=True,
)

def build_bezirk_map_df(predictions: pd.DataFrame) -> pd.DataFrame:
    agg = (
        predictions.groupby("Bezirk")
        .agg(
            erwartet_gesamt=("Erwartet", "sum"),
            max_lift=("Lift", "max"),
            top_kategorie=("Kategorie", lambda x: predictions.loc[x.index].sort_values("Erwartet", ascending=False).iloc[0]["Kategorie"]),
            top_zeitfenster=("Zeitfenster", lambda x: predictions.loc[x.index].sort_values("Erwartet", ascending=False).iloc[0]["Zeitfenster"]),
        )
        .reset_index()
    )
    agg["lat"] = agg["Bezirk"].map(lambda b: BEZIRK_COORDS.get(b, (None, None))[0])
    agg["lon"] = agg["Bezirk"].map(lambda b: BEZIRK_COORDS.get(b, (None, None))[1])
    return agg.dropna(subset=["lat", "lon"])


_TOOLTIP_STYLE = {
    "backgroundColor": "#1a1d2e",
    "color": "white",
    "fontSize": "13px",
    "padding": "10px",
    "borderRadius": "6px",
}


def _bezirk_color_df(map_df: pd.DataFrame) -> pd.DataFrame:
    max_val = map_df["erwartet_gesamt"].max() or 1.0
    max_lift_val = map_df["max_lift"].max() or 1.0
    df = map_df.copy()
    lift_norm = (df["max_lift"] / max_lift_val).clip(0, 1)
    df["r"] = (lift_norm * 220 + 35).astype(int)
    df["g"] = ((1 - lift_norm) * 200 + 20).astype(int)
    df["b"] = 50
    df["elevation"] = (df["erwartet_gesamt"] / max_val * 60000 + 8000).astype(int)
    df["erwartet_fmt"] = df["erwartet_gesamt"].round(3)
    df["max_lift_fmt"] = df["max_lift"].round(2)
    return df


def render_pydeck_choropleth(map_df: pd.DataFrame):
    import json as _json

    geojson_path = Path(__file__).parent.parent / "data" / "berlin_bezirke.geojson"
    if not geojson_path.exists():
        st.warning("berlin_bezirke.geojson nicht gefunden.")
        return

    df = _bezirk_color_df(map_df)
    lookup = {row["Bezirk"]: row for _, row in df.iterrows()}

    with open(geojson_path, encoding="utf-8") as f:
        geojson = _json.load(f)

    for feature in geojson["features"]:
        name = feature["properties"].get("name", "")
        row = lookup.get(name)
        if row is not None:
            feature["Bezirk"] = name
            feature["Erwartet"] = float(row["erwartet_fmt"])
            feature["MaxLift"] = float(row["max_lift_fmt"])
            feature["TopKategorie"] = str(row["top_kategorie"])
            feature["TopZeitfenster"] = str(row["top_zeitfenster"])
            feature["properties"]["fill_color"] = [int(row["r"]), int(row["g"]), int(row["b"]), 210]
        else:
            feature["Bezirk"] = name
            feature["Erwartet"] = 0.0
            feature["MaxLift"] = 0.0
            feature["TopKategorie"] = "-"
            feature["TopZeitfenster"] = "-"
            feature["properties"]["fill_color"] = [60, 60, 70, 160]

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        get_fill_color="properties.fill_color",
        get_line_color=[255, 255, 255, 90],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
        extruded=False,
    )
    view = pdk.ViewState(latitude=52.52, longitude=13.40, zoom=10.0, pitch=0, bearing=0)
    tooltip = {
        "html": (
            "<b>{Bezirk}</b><br/>"
            "Erwartet: {Erwartet}<br/>"
            "Top: {TopKategorie} · {TopZeitfenster}<br/>"
            "Max Lift: {MaxLift}x"
        ),
        "style": _TOOLTIP_STYLE,
    }
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="dark"),
        height=520,
    )


def render_pydeck_columns(map_df: pd.DataFrame):
    df = _bezirk_color_df(map_df)
    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position="[lon, lat]",
        get_elevation="elevation",
        elevation_scale=1,
        radius=2400,
        get_fill_color="[r, g, b, 220]",
        get_line_color=[255, 255, 255, 60],
        pickable=True,
        auto_highlight=True,
        coverage=0.9,
        extruded=True,
    )
    view = pdk.ViewState(latitude=52.52, longitude=13.40, zoom=10.5, pitch=55, bearing=-10)
    tooltip = {
        "html": (
            "<b>{Bezirk}</b><br/>"
            "Erwartet: {erwartet_fmt}<br/>"
            "Top: {top_kategorie} · {top_zeitfenster}<br/>"
            "Max Lift: {max_lift_fmt}x"
        ),
        "style": _TOOLTIP_STYLE,
    }
    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip, map_style="dark"),
        height=580,
    )


def render_folium_map(map_df: pd.DataFrame, predictions: pd.DataFrame):
    m = folium.Map(location=[52.52, 13.40], zoom_start=11, tiles="CartoDB positron")
    max_val = map_df["erwartet_gesamt"].max() or 1.0

    for _, row in map_df.iterrows():
        bezirk = row["Bezirk"]
        radius = int(row["erwartet_gesamt"] / max_val * 18000 + 3000)
        lift_norm = min(row["max_lift"] / 3.0, 1.0)
        r = int(lift_norm * 220 + 35)
        g = int((1 - lift_norm) * 200 + 30)
        color = f"#{r:02x}{g:02x}50"

        top3 = (
            predictions[predictions["Bezirk"] == bezirk]
            .sort_values("Erwartet", ascending=False)
            .head(3)
        )
        popup_rows = "".join(
            f"<tr><td style='padding:2px 6px'>{r2['Zeitfenster']}</td>"
            f"<td style='padding:2px 6px'>{r2['Kategorie']}</td>"
            f"<td style='padding:2px 6px'>{r2['Wahrscheinlichkeit']*100:.1f}%</td></tr>"
            for _, r2 in top3.iterrows()
        )
        popup_html = f"""
        <div style='font-family:sans-serif;min-width:220px'>
            <b style='font-size:14px;color:#111'>{bezirk}</b><br/>
            <span style='color:#555;font-size:12px'>Erwartet gesamt: {row['erwartet_gesamt']:.3f} &nbsp;|&nbsp; Max Lift: {row['max_lift']:.2f}x</span><br/>
            <table style='margin-top:6px;font-size:12px;width:100%;border-collapse:collapse'>
                <tr style='color:#777;border-bottom:1px solid #ddd'><th style='text-align:left;padding:3px'>Fenster</th><th style='text-align:left;padding:3px'>Kategorie</th><th style='text-align:right;padding:3px'>P</th></tr>
                {popup_rows}
            </table>
        </div>"""

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=max(int(row["erwartet_gesamt"] / max_val * 22 + 5), 5),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{bezirk}: {row['top_kategorie']} {row['top_zeitfenster']}",
        ).add_to(m)

    st_folium(m, width="100%", height=520, returned_objects=[])


st.markdown("### Rankings")
tab_abs_30, tab_abs_50, tab_lift = st.tabs(["Top 30 absolut", "Top 50 absolut", "Top 30 Lift"])

with tab_abs_30:
    render_rows(predictions_absolute.head(30), artifacts["exact_df"], mode="absolute")

with tab_abs_50:
    render_rows(predictions_absolute.head(50), artifacts["exact_df"], mode="absolute")

with tab_lift:
    render_rows(predictions_lift.head(30), artifacts["exact_df"], mode="lift")

st.markdown("### Karte")
map_df = build_bezirk_map_df(predictions_absolute)
tab_pydeck, tab_folium = st.tabs(["pydeck", "Folium"])

with tab_pydeck:
    sub_choropleth, sub_columns = st.tabs(["Choropleth", "Säulen (3D)"])
    with sub_choropleth:
        st.caption("Flache Bezirksflächen · Farbe: Rot = hoher Lift, Grün = Basisrate")
        render_pydeck_choropleth(map_df)
    with sub_columns:
        st.caption("3D-Säulen pro Bezirk · Höhe = erwartete Ereignisse · Farbe: Rot = hoher Lift")
        render_pydeck_columns(map_df)

with tab_folium:
    st.caption("Klick auf Bezirk zeigt Top-3 Predictions · Größe = relatives Volumen")
    render_folium_map(map_df, predictions_absolute)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Top Bezirke")
    show_places = top_places.head(15).copy()
    show_places["Erwartet"] = show_places["Erwartet"].round(2)
    st.bar_chart(show_places.set_index("Bezirk")["Erwartet"])

with col2:
    st.markdown("### Top Kategorien")
    show_categories = top_categories.head(15).copy()
    show_categories["Erwartet"] = show_categories["Erwartet"].round(2)
    st.bar_chart(show_categories.set_index("Kategorie")["Erwartet"])
