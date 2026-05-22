import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExpandingMean, RollingMean
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    top_k_accuracy_score,
)
from sklearn.preprocessing import LabelEncoder

from config import COL_PLACE, MODELS_DIR, RANDOM_SEED
from topk_forecasting import (
    TOPK_BUCKET_LABELS,
    add_common_time_features,
    add_time_bucket,
    blend_model_and_prior_probs,
    get_valid_categories,
    load_classified_data,
    poisson_event_probability,
    save_json,
    temporal_split_dates,
)
from topk_forecasting_v2 import (
    apply_lift_cap,
    build_actual_bucket_cells_v2,
    build_category_share_priors,
    build_exact_bucket_dataset,
    get_hierarchical_category_prior,
)


TOPK_V3_MODEL_DIR = MODELS_DIR / "topk_forecaster_v3"
COUNT_MODEL_NAME = "count_cb"
COUNT_LAGS = (1, 7, 14, 28)
LAG_WINDOWS = (7, 14, 28)

_META_PLACE_MARKERS = (
    "/",
    "bundesweit",
    "berlinweit",
    "bezirks",
    "brandenburg",
    "gemeinsame meldung",
    "und brandenburg",
    "unbekannt",
)


def filter_standard_places(df: pd.DataFrame) -> pd.DataFrame:
    mask = ~df[COL_PLACE].astype(str).str.strip().str.lower().apply(
        lambda v: any(m in v for m in _META_PLACE_MARKERS)
    )
    return df[mask].copy()


def add_lag_count_features(
    exact_df: pd.DataFrame,
    windows: tuple[int, ...] = LAG_WINDOWS,
) -> pd.DataFrame:
    """Rolling lag counts per place / bucket / place×bucket. No leakage: shift(1) ensures only past data."""
    df = exact_df.copy()
    df["_date_ts"] = pd.to_datetime(df["date"])

    group_specs = [
        ("lag_place", [COL_PLACE]),
        ("lag_bucket", ["time_bucket"]),
        ("lag_place_bucket", [COL_PLACE, "time_bucket"]),
    ]

    for prefix, gcols in group_specs:
        daily = (
            df.groupby(["_date_ts"] + gcols)
            .size()
            .reset_index(name="_n")
            .sort_values(["_date_ts"] + gcols)
        )
        parts = []
        for key, gdf in daily.groupby(gcols):
            s = gdf.set_index("_date_ts")["_n"].sort_index()
            full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
            s = s.reindex(full_idx, fill_value=0)
            row = s.to_frame("_n").reset_index().rename(columns={"index": "_date_ts"})
            for w in windows:
                row[f"{prefix}_{w}d"] = s.rolling(w, min_periods=0).sum().shift(1).fillna(0).values
            if not isinstance(key, tuple):
                key = (key,)
            for c, v in zip(gcols, key):
                row[c] = v
            parts.append(row[["_date_ts"] + gcols + [f"{prefix}_{w}d" for w in windows]])

        lookup = pd.concat(parts, ignore_index=True)
        df = df.merge(lookup, on=["_date_ts"] + gcols, how="left")
        for w in windows:
            df[f"{prefix}_{w}d"] = df[f"{prefix}_{w}d"].fillna(0).astype(np.float32)

    return df.drop(columns=["_date_ts"])


def build_lag_lookup_for_date(
    df: pd.DataFrame,
    target_date,
    windows: tuple[int, ...] = LAG_WINDOWS,
) -> dict[str, dict]:
    """For prediction: rolling counts from the cache ending at target_date-1."""
    if "time_bucket" not in df.columns:
        df = add_time_bucket(df)
    target_ts = pd.Timestamp(target_date).normalize()
    result: dict[str, dict] = {}
    df_dates = pd.to_datetime(df["date"]).dt.date

    for w in windows:
        window_start = (target_ts - pd.Timedelta(days=w)).date()
        window_end = (target_ts - pd.Timedelta(days=1)).date()
        win = df[(df_dates >= window_start) & (df_dates <= window_end)]
        result[f"lag_place_{w}d"] = win.groupby(COL_PLACE).size().to_dict()
        result[f"lag_bucket_{w}d"] = win.groupby("time_bucket").size().to_dict()
        result[f"lag_place_bucket_{w}d"] = {
            f"{p}|||{b}": int(n)
            for (p, b), n in win.groupby([COL_PLACE, "time_bucket"]).size().items()
        }
    return result


def update_count_forecaster_to_date(
    count_forecaster: MLForecast,
    df: pd.DataFrame,
    data_end_date,
    target_date,
    series_lookup: pd.DataFrame,
) -> MLForecast:
    """Feed actual cache counts into the frozen forecaster so lag features use real data."""
    data_end_ts = pd.Timestamp(data_end_date).normalize()
    target_ts = pd.Timestamp(target_date).normalize()
    gap_dates = pd.date_range(data_end_ts + pd.Timedelta(days=1), target_ts - pd.Timedelta(days=1), freq="D")
    if len(gap_dates) == 0:
        return count_forecaster

    forecaster = copy.deepcopy(count_forecaster)
    gap_df = df[pd.to_datetime(df["date"]).isin(gap_dates)].copy()
    gap_df = add_time_bucket(gap_df)

    counts = (
        gap_df.groupby(["date", COL_PLACE, "time_bucket"])
        .size()
        .reset_index(name="count")
    )
    counts["date"] = pd.to_datetime(counts["date"])

    for day in gap_dates:
        day_counts = counts[counts["date"] == day]
        day_series = series_lookup[[COL_PLACE, "time_bucket", "unique_id"]].copy()
        day_series = day_series.merge(
            day_counts[[COL_PLACE, "time_bucket", "count"]],
            on=[COL_PLACE, "time_bucket"],
            how="left",
        )
        day_series["y"] = day_series["count"].fillna(0).astype(np.float32)
        day_series["ds"] = day
        forecaster.update(
            day_series[["unique_id", "ds", "y", COL_PLACE, "time_bucket"]],
            validate_new_data=False,
        )

    return forecaster


def ensure_topk_v3_model_dir() -> Path:
    TOPK_V3_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return TOPK_V3_MODEL_DIR


def fit_v3_category_encoder(categories: list[str]) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit(np.array(sorted(set(categories)), dtype=str))
    return enc


def build_total_bucket_series(exact_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        exact_df.groupby(["date", COL_PLACE, "time_bucket"])
        .size()
        .reset_index(name="count")
    )

    all_dates = pd.date_range(exact_df["date"].min(), exact_df["date"].max(), freq="D")
    all_places = sorted(exact_df[COL_PLACE].astype(str).unique().tolist())
    grid = pd.MultiIndex.from_product(
        [all_dates, all_places, list(TOPK_BUCKET_LABELS)],
        names=["ds", COL_PLACE, "time_bucket"],
    ).to_frame(index=False)
    grid["date"] = grid["ds"].dt.date

    merged = grid.merge(counts, on=["date", COL_PLACE, "time_bucket"], how="left")
    merged["count"] = merged["count"].fillna(0).astype(np.int16)
    merged["unique_id"] = (
        merged[COL_PLACE].astype(str)
        + "|||"
        + merged["time_bucket"].astype(str)
    )
    merged["y"] = merged["count"].astype(np.float32)
    return merged[
        ["unique_id", "ds", "y", COL_PLACE, "time_bucket", "date", "count"]
    ].copy()


def build_series_lookup(total_series: pd.DataFrame) -> pd.DataFrame:
    return (
        total_series[["unique_id", COL_PLACE, "time_bucket"]]
        .drop_duplicates()
        .sort_values([COL_PLACE, "time_bucket"])
        .reset_index(drop=True)
    )


def make_count_forecaster(model: CatBoostRegressor | None = None) -> MLForecast:
    if model is None:
        model = CatBoostRegressor(
            loss_function="Poisson",
            eval_metric="Poisson",
            iterations=600,
            learning_rate=0.045,
            depth=6,
            l2_leaf_reg=6.0,
            min_data_in_leaf=20,
            random_seed=RANDOM_SEED,
            verbose=False,
            allow_writing_files=False,
            has_time=True,
            thread_count=-1,
            cat_features=(COL_PLACE, "time_bucket"),
        )

    return MLForecast(
        models={COUNT_MODEL_NAME: model},
        freq="D",
        lags=list(COUNT_LAGS),
        lag_transforms={
            1: [RollingMean(3), RollingMean(7), ExpandingMean()],
            7: [RollingMean(4)],
            28: [RollingMean(2)],
        },
        date_features=["dayofweek", "month", "day", "dayofyear", "quarter"],
        num_threads=1,
    )


def fit_count_forecaster(total_series: pd.DataFrame) -> MLForecast:
    forecaster = make_count_forecaster()
    fit_df = total_series[["unique_id", "ds", "y", COL_PLACE, "time_bucket"]].copy()
    forecaster.fit(
        fit_df,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
        static_features=[COL_PLACE, "time_bucket"],
    )
    return forecaster


def build_share_feature_frame(
    df: pd.DataFrame,
    lag_lookup: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    dt = pd.to_datetime(df["date"])
    weekday = df["weekday"].astype(np.float32)
    month = df["month"].astype(np.float32)
    dayofyear = df["dayofyear"].astype(np.float32)
    iso_week = dt.dt.isocalendar().week.astype(np.int16)

    features = pd.DataFrame({
        COL_PLACE: df[COL_PLACE].astype(str),
        "time_bucket": df["time_bucket"].astype(str),
        "place_bucket": (
            df[COL_PLACE].astype(str)
            + "|||"
            + df["time_bucket"].astype(str)
        ),
        "weekday_sin": np.sin(2 * np.pi * weekday / 7).astype(np.float32),
        "weekday_cos": np.cos(2 * np.pi * weekday / 7).astype(np.float32),
        "month_sin": np.sin(2 * np.pi * month / 12).astype(np.float32),
        "month_cos": np.cos(2 * np.pi * month / 12).astype(np.float32),
        "dayofyear_sin": np.sin(2 * np.pi * dayofyear / 366).astype(np.float32),
        "dayofyear_cos": np.cos(2 * np.pi * dayofyear / 366).astype(np.float32),
        "weekday": df["weekday"].astype(np.int8),
        "weekofyear": iso_week,
        "month": df["month"].astype(np.int8),
        "day": df["day"].astype(np.int8),
        "quarter": df["quarter"].astype(np.int8),
        "year": df["year"].astype(np.int16),
        "is_weekend": df["is_weekend"].astype(np.int8),
        "days_since_start": df["days_since_start"].astype(np.int32),
        "month_index": df["month_index"].astype(np.int16),
    })

    pb_key = df[COL_PLACE].astype(str) + "|||" + df["time_bucket"].astype(str)
    for w in LAG_WINDOWS:
        if lag_lookup is not None:
            features[f"lag_place_{w}d"] = (
                df[COL_PLACE].astype(str).map(lag_lookup.get(f"lag_place_{w}d", {})).fillna(0).astype(np.float32)
            )
            features[f"lag_bucket_{w}d"] = (
                df["time_bucket"].astype(str).map(lag_lookup.get(f"lag_bucket_{w}d", {})).fillna(0).astype(np.float32)
            )
            features[f"lag_place_bucket_{w}d"] = (
                pb_key.map(lag_lookup.get(f"lag_place_bucket_{w}d", {})).fillna(0).astype(np.float32)
            )
        else:
            for prefix in ("lag_place", "lag_bucket", "lag_place_bucket"):
                col = f"{prefix}_{w}d"
                features[col] = df[col].astype(np.float32) if col in df.columns else np.float32(0.0)

    return features, [COL_PLACE, "time_bucket", "place_bucket"]


def make_share_model(cat_features: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        iterations=700,
        learning_rate=0.045,
        depth=8,
        l2_leaf_reg=6.0,
        min_data_in_leaf=20,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        has_time=True,
        thread_count=-1,
        cat_features=tuple(cat_features),
        od_type="Iter",
        od_wait=60,
    )


def fit_share_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: list[str],
) -> CatBoostClassifier:
    model = make_share_model(cat_features)
    train_pool = Pool(X_train, label=y_train, cat_features=cat_features)
    val_pool = Pool(X_val, label=y_val, cat_features=cat_features)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    return model


def evaluate_share_model(
    model: CatBoostClassifier,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    num_classes: int,
) -> dict:
    y_pred = np.asarray(model.predict(X_val)).astype(int).reshape(-1)
    y_proba = np.asarray(model.predict_proba(X_val))
    labels = list(range(num_classes))
    return {
        "accuracy_top1": float(accuracy_score(y_val, y_pred)),
        "accuracy_top3": float(top_k_accuracy_score(y_val, y_proba, k=min(3, num_classes), labels=labels)),
        "accuracy_top5": float(top_k_accuracy_score(y_val, y_proba, k=min(5, num_classes), labels=labels)),
        "f1_macro": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y_val, y_proba, labels=labels)),
        "n_samples": int(len(y_val)),
    }


def build_prediction_cells(
    target_date,
    series_lookup: pd.DataFrame,
    data_start_date,
) -> pd.DataFrame:
    rows = series_lookup[[COL_PLACE, "time_bucket"]].copy()
    rows["datetime"] = pd.Timestamp(target_date)
    return add_common_time_features(rows, data_start_date=data_start_date)


def _get_forecast_value_column(forecast_df: pd.DataFrame) -> str:
    for col in forecast_df.columns:
        if col not in {"unique_id", "ds"}:
            return col
    raise ValueError("Forecast-Spalte nicht gefunden.")


def forecast_count_cells(
    count_forecaster: MLForecast,
    target_date,
    history_end_date,
    series_lookup: pd.DataFrame,
) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_date).normalize()
    history_end_ts = pd.Timestamp(history_end_date).normalize()
    horizon = int((target_ts - history_end_ts).days)
    if horizon < 1:
        raise ValueError(
            f"V3 unterstuetzt nur Prognosen nach dem Trainingsende ({history_end_ts.date()})."
        )

    forecast_df = count_forecaster.predict(h=horizon)
    value_col = _get_forecast_value_column(forecast_df)
    target_rows = forecast_df[forecast_df["ds"] == target_ts].copy()
    target_rows = target_rows.merge(series_lookup, on="unique_id", how="left")
    target_rows["count_pred"] = np.clip(target_rows[value_col], 0.0, None)
    return target_rows[["unique_id", "ds", COL_PLACE, "time_bucket", "count_pred"]]


def build_rankings_from_count_frame(
    target_date,
    count_frame: pd.DataFrame,
    share_model: CatBoostClassifier,
    cat_enc: LabelEncoder,
    priors: dict,
    metadata: dict,
    series_lookup: pd.DataFrame,
    sort_by: str = "absolute",
    max_per_category: int | None = None,
    lag_lookup: dict | None = None,
) -> pd.DataFrame:
    pred_cells = build_prediction_cells(
        target_date=target_date,
        series_lookup=series_lookup,
        data_start_date=metadata["data_start_date"],
    )
    pred_cells = pred_cells.merge(
        count_frame[[COL_PLACE, "time_bucket", "count_pred"]],
        on=[COL_PLACE, "time_bucket"],
        how="left",
    )
    pred_cells["count_pred"] = pred_cells["count_pred"].fillna(0.0)

    X_share_pred, _ = build_share_feature_frame(pred_cells, lag_lookup=lag_lookup)
    share_probs_model = np.asarray(share_model.predict_proba(X_share_pred))
    categories = cat_enc.classes_.tolist()
    prior_weight = float(metadata.get("prior_weight", 0.6))
    max_lift = metadata.get("max_lift")

    rows = []
    for idx, base_row in pred_cells[[COL_PLACE, "time_bucket", "count_pred"]].iterrows():
        place = str(base_row[COL_PLACE])
        bucket = str(base_row["time_bucket"])
        prior_probs = get_hierarchical_category_prior(priors, place, bucket)
        final_probs = blend_model_and_prior_probs(
            model_probs=share_probs_model[idx],
            prior_probs=prior_probs,
            prior_weight=prior_weight,
        )
        final_probs = apply_lift_cap(final_probs, prior_probs, max_lift)

        total_lam = float(base_row["count_pred"])
        total_prob = poisson_event_probability(total_lam)
        for cat_idx, category in enumerate(categories):
            share = float(final_probs[cat_idx])
            base_share = float(prior_probs[cat_idx])
            lam = total_lam * share
            prob = poisson_event_probability(lam)
            lift = share / max(base_share, 1e-9)
            rows.append({
                "Bezirk": place,
                "Zeitfenster": bucket,
                "Kategorie": category,
                "Zellenwert": total_lam,
                "Zellenwahrscheinlichkeit": total_prob,
                "Anteil": share,
                "BasisAnteil": base_share,
                "Lift": lift,
                "LiftScore": prob * max(lift, 1e-9),
                "Erwartet": lam,
                "Wahrscheinlichkeit": prob,
            })

    result = pd.DataFrame(rows)
    if sort_by == "lift":
        result = result.sort_values(
            ["LiftScore", "Lift", "Wahrscheinlichkeit", "Erwartet"],
            ascending=False,
        )
    else:
        result = result.sort_values(
            ["Wahrscheinlichkeit", "Erwartet", "Anteil"],
            ascending=False,
        )
    result = result.reset_index(drop=True)

    if max_per_category is not None and max_per_category > 0:
        counts: dict[str, int] = {}
        keep = []
        for idx, row in result.iterrows():
            cat = str(row["Kategorie"])
            n = counts.get(cat, 0)
            if n < max_per_category:
                counts[cat] = n + 1
                keep.append(idx)
        result = result.loc[keep].reset_index(drop=True)

    return result


def predict_v3_rankings(
    target_date,
    artifacts: dict,
    sort_by: str = "absolute",
    max_per_category: int | None = None,
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    metadata = artifacts["metadata"]
    series_lookup = artifacts["series_lookup"]

    if df is not None:
        count_forecaster = update_count_forecaster_to_date(
            count_forecaster=artifacts["count_forecaster"],
            df=df,
            data_end_date=metadata["data_end_date"],
            target_date=target_date,
            series_lookup=series_lookup,
        )
        lag_lookup = build_lag_lookup_for_date(df, target_date)
    else:
        count_forecaster = artifacts["count_forecaster"]
        lag_lookup = None

    count_frame = forecast_count_cells(
        count_forecaster=count_forecaster,
        target_date=target_date,
        history_end_date=metadata["data_end_date"],
        series_lookup=series_lookup,
    )
    return build_rankings_from_count_frame(
        target_date=target_date,
        count_frame=count_frame,
        share_model=artifacts["share_model"],
        cat_enc=artifacts["cat_enc"],
        priors=artifacts["priors"],
        metadata=metadata,
        series_lookup=series_lookup,
        sort_by=sort_by,
        max_per_category=max_per_category,
        lag_lookup=lag_lookup,
    )


def evaluate_count_backtest(
    val_dates: list,
    val_total_series: pd.DataFrame,
    count_forecaster: MLForecast,
    series_lookup: pd.DataFrame,
) -> dict:
    forecaster = copy.deepcopy(count_forecaster)
    pred_values = []
    true_values = []

    for val_date in val_dates:
        forecast_df = forecaster.predict(h=1)
        value_col = _get_forecast_value_column(forecast_df)
        pred_day = forecast_df.merge(series_lookup, on="unique_id", how="left")
        pred_day["count_pred"] = np.clip(pred_day[value_col], 0.0, None)

        actual_day = val_total_series[val_total_series["date"] == val_date][
            ["unique_id", "y", COL_PLACE, "time_bucket", "ds"]
        ].copy()
        actual_day["y"] = actual_day["y"].astype(float)
        merged = actual_day.merge(
            pred_day[["unique_id", "count_pred"]],
            on="unique_id",
            how="left",
        )
        merged["count_pred"] = merged["count_pred"].fillna(0.0)
        pred_values.extend(merged["count_pred"].tolist())
        true_values.extend(merged["y"].tolist())
        forecaster.update(
            actual_day.rename(columns={"y": "y"}),
            validate_new_data=False,
        )

    y_true = np.asarray(true_values, dtype=np.float64)
    y_pred = np.asarray(pred_values, dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None))),
        "mean_true": float(np.mean(y_true)),
        "mean_pred": float(np.mean(y_pred)),
        "sum_true": float(np.sum(y_true)),
        "sum_pred": float(np.sum(y_pred)),
    }


def evaluate_topk_rankings_v3(
    val_dates: list,
    actual_cells: pd.DataFrame,
    val_total_series: pd.DataFrame,
    artifacts: dict,
    ks=(30, 50),
    sort_by: str = "absolute",
) -> dict:
    metrics = {f"precision_at_{k}": [] for k in ks}
    metrics.update({f"recall_at_{k}": [] for k in ks})
    metrics["days_with_exact_events"] = 0

    forecaster = copy.deepcopy(artifacts["count_forecaster"])
    series_lookup = artifacts["series_lookup"]

    for val_date in val_dates:
        actual_day = actual_cells[actual_cells["date"] == val_date]
        forecast_df = forecaster.predict(h=1)
        value_col = _get_forecast_value_column(forecast_df)
        count_frame = forecast_df.merge(series_lookup, on="unique_id", how="left")
        count_frame["count_pred"] = np.clip(count_frame[value_col], 0.0, None)

        if len(actual_day) > 0:
            metrics["days_with_exact_events"] += 1
            actual_set = {
                (str(r[COL_PLACE]), str(r["category"]), str(r["time_bucket"]))
                for _, r in actual_day.iterrows()
            }
            pred = build_rankings_from_count_frame(
                target_date=val_date,
                count_frame=count_frame,
                share_model=artifacts["share_model"],
                cat_enc=artifacts["cat_enc"],
                priors=artifacts["priors"],
                metadata=artifacts["metadata"],
                series_lookup=series_lookup,
                sort_by=sort_by,
            )
            for k in ks:
                top = pred.head(k)
                pred_set = {
                    (str(r["Bezirk"]), str(r["Kategorie"]), str(r["Zeitfenster"]))
                    for _, r in top.iterrows()
                }
                hits = len(actual_set & pred_set)
                metrics[f"precision_at_{k}"].append(hits / k)
                metrics[f"recall_at_{k}"].append(hits / max(len(actual_set), 1))

        update_day = val_total_series[val_total_series["date"] == val_date][
            ["unique_id", "ds", "y", COL_PLACE, "time_bucket"]
        ].copy()
        forecaster.update(update_day, validate_new_data=False)

    summary = {
        "days_with_exact_events": int(metrics["days_with_exact_events"]),
    }
    for k in ks:
        values_precision = metrics[f"precision_at_{k}"]
        values_recall = metrics[f"recall_at_{k}"]
        summary[f"precision_at_{k}"] = float(np.mean(values_precision)) if values_precision else 0.0
        summary[f"recall_at_{k}"] = float(np.mean(values_recall)) if values_recall else 0.0
    return summary


def save_count_feature_importance(count_forecaster: MLForecast, path: Path):
    model = count_forecaster.models_[COUNT_MODEL_NAME]
    df = model.get_feature_importance(prettified=True)
    df = df.rename(columns={"Feature Id": "feature", "Importances": "importance"})
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_share_feature_importance(share_model: CatBoostClassifier, path: Path):
    df = share_model.get_feature_importance(prettified=True)
    df = df.rename(columns={"Feature Id": "feature", "Importances": "importance"})
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_topk_v3_artifacts(include_data: bool = True) -> dict:
    model_dir = TOPK_V3_MODEL_DIR
    count_path = model_dir / "count_forecaster.pkl"
    share_path = model_dir / "category_share_model.cbm"
    enc_path = model_dir / "encoders.pkl"
    meta_path = model_dir / "metadata.json"
    prior_path = model_dir / "category_share_priors.json"
    eval_path = model_dir / "evaluation.json"
    series_lookup_path = model_dir / "series_lookup.csv"

    for path in [count_path, share_path, enc_path, meta_path, prior_path, series_lookup_path]:
        if not path.exists():
            raise FileNotFoundError(f"Fehlt: {path}")

    with open(count_path, "rb") as f:
        count_forecaster = pickle.load(f)

    share_model = CatBoostClassifier()
    share_model.load_model(share_path)

    with open(enc_path, "rb") as f:
        encoders = pickle.load(f)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    priors = json.loads(prior_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    series_lookup = pd.read_csv(series_lookup_path, encoding="utf-8-sig")

    artifacts = {
        "count_forecaster": count_forecaster,
        "share_model": share_model,
        "cat_enc": encoders["cat_enc"],
        "metadata": metadata,
        "priors": priors,
        "evaluation": evaluation,
        "series_lookup": series_lookup,
    }

    if include_data:
        artifacts["df"] = load_classified_data()

    return artifacts


__all__ = [
    "COUNT_LAGS",
    "COUNT_MODEL_NAME",
    "LAG_WINDOWS",
    "TOPK_BUCKET_LABELS",
    "TOPK_V3_MODEL_DIR",
    "add_lag_count_features",
    "build_actual_bucket_cells_v2",
    "build_lag_lookup_for_date",
    "filter_standard_places",
    "update_count_forecaster_to_date",
    "build_category_share_priors",
    "build_exact_bucket_dataset",
    "build_prediction_cells",
    "build_rankings_from_count_frame",
    "build_series_lookup",
    "build_share_feature_frame",
    "build_total_bucket_series",
    "ensure_topk_v3_model_dir",
    "evaluate_count_backtest",
    "evaluate_share_model",
    "evaluate_topk_rankings_v3",
    "fit_count_forecaster",
    "fit_share_model",
    "fit_v3_category_encoder",
    "forecast_count_cells",
    "get_valid_categories",
    "load_classified_data",
    "load_topk_v3_artifacts",
    "make_count_forecaster",
    "make_share_model",
    "poisson_event_probability",
    "predict_v3_rankings",
    "save_count_feature_importance",
    "save_json",
    "save_share_feature_importance",
    "temporal_split_dates",
]
