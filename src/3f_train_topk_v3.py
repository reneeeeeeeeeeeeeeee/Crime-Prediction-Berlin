"""
SCHRITT 3f: Top-K Forecasting V3

Stack:
  1. MLForecast + CatBoostRegressor fuer count_total(date, place, bucket)
  2. CatBoostClassifier fuer P(category | date, place, bucket)
  3. Prior-Blending + Lift-Cap fuer stabile Rankings
"""
import argparse
import logging
import pickle
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from catboost import Pool

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RANDOM_SEED
from topk_forecasting_v3 import (
    TOPK_BUCKET_LABELS,
    add_lag_count_features,
    build_actual_bucket_cells_v2,
    build_category_share_priors,
    build_share_feature_frame,
    build_total_bucket_series,
    build_series_lookup,
    build_exact_bucket_dataset,
    ensure_topk_v3_model_dir,
    evaluate_count_backtest,
    evaluate_share_model,
    evaluate_topk_rankings_v3,
    filter_standard_places,
    fit_count_forecaster,
    fit_share_model,
    fit_v3_category_encoder,
    get_valid_categories,
    load_classified_data,
    make_share_model,
    predict_v3_rankings,
    save_count_feature_importance,
    save_json,
    save_share_feature_importance,
    temporal_split_dates,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PRIOR_WEIGHT = 0.60
DEFAULT_MAX_LIFT = 2.75


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--mlflow-experiment", type=str, default="topk_forecasting_v3")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


@contextmanager
def optional_mlflow_run(experiment_name: str | None, enabled: bool):
    if not enabled or not experiment_name:
        yield None
        return

    try:
        import mlflow
    except Exception as exc:
        logger.warning("MLflow nicht verfuegbar: %s", exc)
        yield None
        return

    tracking_dir = Path(__file__).resolve().parent.parent / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="topk_forecasting_v3"):
        yield mlflow


def optimize_postprocessing(
    val_dates: list,
    actual_cells: pd.DataFrame,
    val_total_series: pd.DataFrame,
    backtest_artifacts: dict,
    n_trials: int,
) -> tuple[float, float, dict]:
    if n_trials <= 0:
        return DEFAULT_PRIOR_WEIGHT, DEFAULT_MAX_LIFT, {}

    import optuna

    logger.info("Starte Optuna fuer Prior-Blending (%s Trials) ...", n_trials)

    def objective(trial):
        trial_artifacts = dict(backtest_artifacts)
        metadata = dict(backtest_artifacts["metadata"])
        metadata["prior_weight"] = trial.suggest_float("prior_weight", 0.35, 0.80)
        metadata["max_lift"] = trial.suggest_float("max_lift", 1.50, 4.00)
        trial_artifacts["metadata"] = metadata

        metrics = evaluate_topk_rankings_v3(
            val_dates=val_dates,
            actual_cells=actual_cells,
            val_total_series=val_total_series,
            artifacts=trial_artifacts,
            ks=(30, 50),
            sort_by="absolute",
        )
        score = (
            metrics["recall_at_30"]
            + metrics["recall_at_50"]
            + 0.5 * metrics["precision_at_30"]
        )
        trial.set_user_attr("recall_at_30", metrics["recall_at_30"])
        trial.set_user_attr("recall_at_50", metrics["recall_at_50"])
        trial.set_user_attr("precision_at_30", metrics["precision_at_30"])
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    summary = {
        "best_value": float(study.best_value),
        "best_params": {
            "prior_weight": float(study.best_params["prior_weight"]),
            "max_lift": float(study.best_params["max_lift"]),
        },
        "n_trials": int(len(study.trials)),
    }
    return (
        summary["best_params"]["prior_weight"],
        summary["best_params"]["max_lift"],
        summary,
    )


def fit_full_share_model(X_full, y_full, cat_features: list[str]):
    model = make_share_model(cat_features)
    full_pool = Pool(X_full, label=y_full, cat_features=cat_features)
    model.fit(full_pool, use_best_model=False)
    return model


def main():
    args = parse_args()

    logger.info("=" * 72)
    logger.info("SCHRITT 3f: Top-K Forecasting V3")
    logger.info("=" * 72)

    model_dir = ensure_topk_v3_model_dir()

    with optional_mlflow_run(
        experiment_name=args.mlflow_experiment,
        enabled=not args.disable_mlflow,
    ) as mlflow:
        df = load_classified_data()
        logger.info("Geladene Meldungen: %s", f"{len(df):,}")

        df = filter_standard_places(df)
        logger.info("Nach Meta-Place-Filter: %s", f"{len(df):,}")

        valid_categories = get_valid_categories(df, min_events=10)
        exact_df = build_exact_bucket_dataset(df, valid_categories)
        exact_df = add_lag_count_features(exact_df)
        logger.info("Lag-Features berechnet (7/14/28 Tage fuer place, bucket, place×bucket)")
        cat_enc = fit_v3_category_encoder(valid_categories)
        logger.info("Gueltige Kategorien (>=10 Events): %s", len(valid_categories))
        logger.info("Exakte Zeitfaelle fuer V3: %s", f"{len(exact_df):,}")

        total_series = build_total_bucket_series(exact_df)
        series_lookup = build_series_lookup(total_series)
        dates = sorted(total_series["date"].unique())
        train_dates, val_dates = temporal_split_dates(dates, val_ratio=0.2)

        train_total_series = total_series[total_series["date"].isin(train_dates)].copy()
        val_total_series = total_series[total_series["date"].isin(val_dates)].copy()
        train_share_df = exact_df[exact_df["date"].isin(train_dates)].copy()
        val_share_df = exact_df[exact_df["date"].isin(val_dates)].copy()

        logger.info("Train-Zeitraum: %s -> %s (%s Tage)", train_dates[0], train_dates[-1], len(train_dates))
        logger.info("Val-Zeitraum:   %s -> %s (%s Tage)", val_dates[0], val_dates[-1], len(val_dates))
        logger.info("Count-Serien: %s", series_lookup["unique_id"].nunique())

        logger.info("Trainiere Backtest-count_total-Modell (MLForecast + CatBoost) ...")
        backtest_count_forecaster = fit_count_forecaster(train_total_series)
        count_metrics = evaluate_count_backtest(
            val_dates=val_dates,
            val_total_series=val_total_series,
            count_forecaster=backtest_count_forecaster,
            series_lookup=series_lookup,
        )

        logger.info("Trainiere Backtest-category_share-Modell (CatBoost) ...")
        X_share_train, share_cat_features = build_share_feature_frame(train_share_df)
        X_share_val, _ = build_share_feature_frame(val_share_df)
        y_share_train = cat_enc.transform(train_share_df["category"].astype(str))
        y_share_val = cat_enc.transform(val_share_df["category"].astype(str))
        backtest_share_model = fit_share_model(
            X_train=X_share_train,
            y_train=y_share_train,
            X_val=X_share_val,
            y_val=y_share_val,
            cat_features=share_cat_features,
        )
        share_metrics = evaluate_share_model(
            backtest_share_model,
            X_share_val,
            y_share_val,
            num_classes=len(cat_enc.classes_),
        )

        backtest_priors = build_category_share_priors(
            train_share_df,
            categories=cat_enc.classes_.tolist(),
            bucket_labels=list(TOPK_BUCKET_LABELS),
        )
        backtest_artifacts = {
            "count_forecaster": backtest_count_forecaster,
            "share_model": backtest_share_model,
            "cat_enc": cat_enc,
            "priors": backtest_priors,
            "series_lookup": series_lookup,
            "metadata": {
                "data_start_date": str(total_series["date"].min()),
                "data_end_date": str(train_dates[-1]),
                "bucket_labels": list(TOPK_BUCKET_LABELS),
                "prior_weight": DEFAULT_PRIOR_WEIGHT,
                "max_lift": DEFAULT_MAX_LIFT,
            },
        }

        actual_cells = build_actual_bucket_cells_v2(val_share_df)
        tuned_prior_weight, tuned_max_lift, optuna_summary = optimize_postprocessing(
            val_dates=val_dates,
            actual_cells=actual_cells,
            val_total_series=val_total_series,
            backtest_artifacts=backtest_artifacts,
            n_trials=args.optuna_trials,
        )
        backtest_artifacts["metadata"]["prior_weight"] = tuned_prior_weight
        backtest_artifacts["metadata"]["max_lift"] = tuned_max_lift

        topk_metrics = evaluate_topk_rankings_v3(
            val_dates=val_dates,
            actual_cells=actual_cells,
            val_total_series=val_total_series,
            artifacts=backtest_artifacts,
            ks=(30, 50),
            sort_by="absolute",
        )

        logger.info("Trainiere finale V3-Modelle auf allen Daten ...")
        final_count_forecaster = fit_count_forecaster(total_series)
        X_share_full, share_cat_features = build_share_feature_frame(exact_df)
        y_share_full = cat_enc.transform(exact_df["category"].astype(str))
        final_share_model = fit_full_share_model(X_share_full, y_share_full, share_cat_features)
        final_priors = build_category_share_priors(
            exact_df,
            categories=cat_enc.classes_.tolist(),
            bucket_labels=list(TOPK_BUCKET_LABELS),
        )

        metadata = {
            "architecture": "mlforecast_catboost_count_plus_catboost_share",
            "frameworks": ["mlforecast", "catboost"],
            "count_feature_lags": [1, 7, 14, 28],
            "data_start_date": str(total_series["date"].min()),
            "data_end_date": str(total_series["date"].max()),
            "train_start_date": str(train_dates[0]),
            "train_end_date": str(train_dates[-1]),
            "val_start_date": str(val_dates[0]),
            "val_end_date": str(val_dates[-1]),
            "n_events": int(len(df)),
            "n_exact_events": int(len(exact_df)),
            "exact_time_share": float((df["time_confidence"] == 0).mean()),
            "n_valid_categories": int(len(valid_categories)),
            "n_places": int(series_lookup["place"].nunique()),
            "n_place_bucket_series": int(series_lookup["unique_id"].nunique()),
            "bucket_labels": list(TOPK_BUCKET_LABELS),
            "prior_weight": float(tuned_prior_weight),
            "max_lift": float(tuned_max_lift),
            "ranking_modes": ["absolute", "lift"],
            "train_on_exact_times_only": True,
            "random_seed": RANDOM_SEED,
        }

        evaluation = {
            "count_model": count_metrics,
            "share_model": share_metrics,
            "topk_backtest": topk_metrics,
            "optuna": optuna_summary,
        }

        logger.info("Speichere V3-Artefakte ...")
        with open(model_dir / "count_forecaster.pkl", "wb") as f:
            pickle.dump(final_count_forecaster, f)

        final_share_model.save_model(model_dir / "category_share_model.cbm")
        with open(model_dir / "encoders.pkl", "wb") as f:
            pickle.dump({"cat_enc": cat_enc}, f)

        series_lookup.to_csv(model_dir / "series_lookup.csv", index=False, encoding="utf-8-sig")
        save_json(model_dir / "metadata.json", metadata)
        save_json(model_dir / "evaluation.json", evaluation)
        save_json(model_dir / "category_share_priors.json", final_priors)
        if optuna_summary:
            save_json(model_dir / "optuna_summary.json", optuna_summary)

        save_count_feature_importance(final_count_forecaster, model_dir / "count_feature_importance.csv")
        save_share_feature_importance(final_share_model, model_dir / "share_feature_importance.csv")

        final_artifacts = {
            "count_forecaster": final_count_forecaster,
            "share_model": final_share_model,
            "cat_enc": cat_enc,
            "priors": final_priors,
            "series_lookup": series_lookup,
            "metadata": metadata,
        }
        preview_date = (pd.Timestamp(metadata["data_end_date"]) + pd.Timedelta(days=1)).date()
        preview_absolute = predict_v3_rankings(preview_date, final_artifacts, sort_by="absolute").head(30)
        preview_lift = predict_v3_rankings(preview_date, final_artifacts, sort_by="lift").head(30)
        preview_absolute.to_csv(model_dir / "preview_top30_absolute.csv", index=False, encoding="utf-8-sig")
        preview_lift.to_csv(model_dir / "preview_top30_lift.csv", index=False, encoding="utf-8-sig")

        if mlflow is not None:
            mlflow.log_params({
                "architecture": metadata["architecture"],
                "n_valid_categories": metadata["n_valid_categories"],
                "n_places": metadata["n_places"],
                "prior_weight": metadata["prior_weight"],
                "max_lift": metadata["max_lift"],
                "optuna_trials": int(args.optuna_trials),
            })
            mlflow.log_metrics({
                "count_mae": count_metrics["mae"],
                "count_poisson_deviance": count_metrics["poisson_deviance"],
                "share_accuracy_top1": share_metrics["accuracy_top1"],
                "share_accuracy_top3": share_metrics["accuracy_top3"],
                "topk_recall_at_30": topk_metrics["recall_at_30"],
                "topk_recall_at_50": topk_metrics["recall_at_50"],
                "topk_precision_at_30": topk_metrics["precision_at_30"],
                "topk_precision_at_50": topk_metrics["precision_at_50"],
            })
            mlflow.log_artifacts(str(model_dir))

        logger.info("=" * 72)
        logger.info("FERTIG")
        logger.info("Artefakte: %s", model_dir)
        logger.info("Backtest Recall@30: %.4f", topk_metrics["recall_at_30"])
        logger.info("Backtest Recall@50: %.4f", topk_metrics["recall_at_50"])
        logger.info("=" * 72)


if __name__ == "__main__":
    main()
