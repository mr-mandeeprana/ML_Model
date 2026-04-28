"""
Run ML Standalone Pipeline
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
 
import pandas as pd
 
# Add project root to sys.path for standalone execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
 
from ml_model.data_preprocessing import DataPreprocessor
from ml_model.feature_engineering import FeatureEngineer
from ml_model.model_training import MLTrainer
from ml_model.predictor import MLPredictor
 
 
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML standalone training pipeline.")
    parser.add_argument(
        "--raw-data",
        default="data/data_25_transformed.csv",
        help="Path to raw long-format sensor CSV.",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Use strict ML prediction output (no calendar blending/fallback in model-ready path).",
    )
    parser.add_argument(
        "--prediction-window",
        type=int,
        default=100,
        help="Number of latest feature rows used for final prediction report.",
    )
    return parser.parse_args()


def _print_data_quality_and_degradation_report(cleaned_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
    cleaned_df = cleaned_df.copy()
    features_df = features_df.copy()

    cleaned_df["@timestamp"] = pd.to_datetime(cleaned_df["@timestamp"], utc=True, errors="coerce")
    features_df["@timestamp"] = pd.to_datetime(features_df["@timestamp"], utc=True, errors="coerce")

    cleaned_df = cleaned_df.dropna(subset=["@timestamp"])
    features_df = features_df.dropna(subset=["@timestamp"])

    total_rows = len(cleaned_df)
    total_minutes = max(cleaned_df["@timestamp"].nunique(), 1)
    expected_rows = total_minutes * len(DataPreprocessor.CORE_SENSORS)
    coverage_pct = (100.0 * total_rows / expected_rows) if expected_rows else 0.0

    sensor_coverage = (
        cleaned_df.groupby("sensorid")["@timestamp"].nunique().sort_index()
        / total_minutes
        * 100.0
    )

    window = max(len(features_df) // 10, 1)
    early_window = features_df.head(window)
    late_window = features_df.tail(window)

    degradation_early = float(early_window.get("degradation_index", pd.Series([0.0])).mean())
    degradation_late = float(late_window.get("degradation_index", pd.Series([0.0])).mean())
    degradation_delta = degradation_late - degradation_early
    rising_share = float(features_df.get("degradation_rising_flag", pd.Series([0.0])).mean()) * 100.0

    warning_total = int(features_df.get("warning_count", pd.Series([0])).sum())
    critical_total = int(features_df.get("critical_count", pd.Series([0])).sum())

    temp_warning_share = float(features_df.get("temp_warning", pd.Series([0])).mean()) * 100.0
    elong_warning_share = float(features_df.get("elong_warning", pd.Series([0])).mean()) * 100.0
    current_warning_share = float(features_df.get("current_warning", pd.Series([0])).mean()) * 100.0

    print("\n" + "=" * 90)
    print("DATA QUALITY AND DEGRADATION SUMMARY")
    print("=" * 90)
    print(f"Rows after preprocessing: {total_rows}")
    print(f"Timestamp coverage: {coverage_pct:.2f}% of the expected minute-level rows")
    print("Sensor uptime:")
    for sensor, pct in sensor_coverage.items():
        print(f"  - {sensor}: {pct:.2f}%")
    print(f"Warning events: {warning_total}")
    print(f"Critical events: {critical_total}")
    print(f"Temp warning share: {temp_warning_share:.2f}%")
    print(f"Elongation warning share: {elong_warning_share:.2f}%")
    print(f"Current warning share: {current_warning_share:.2f}%")
    print(f"Degradation index start mean: {degradation_early:.4f}")
    print(f"Degradation index end mean:   {degradation_late:.4f}")
    print(f"Degradation trend delta:      {degradation_delta:.4f}")
    print(f"Rising degradation share:     {rising_share:.2f}%")
    print("=" * 90)


def run_ml_pipeline(
    raw_data_path_str: str | None = None,
    ml_only: bool = False,
    prediction_window: int = 100,
) -> None:
    base_dir = ROOT_DIR
 
    if raw_data_path_str:
        raw_data_path = Path(raw_data_path_str)
    else:
        args = _parse_args()
        raw_data_path = Path(args.raw_data)
        
    if not raw_data_path.is_absolute():
        raw_data_path = base_dir / raw_data_path
 
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_data_path}")

    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = processed_dir / "cleaned_sensor_data.csv"
    features_path = processed_dir / "features.csv"

    print("=" * 90)
    print("STEP 1/4 - PREPROCESSING RAW SENSOR DATA")
    print("=" * 90)

    preprocessor = DataPreprocessor(
        config_path=str(base_dir / "config" / "thresholds.json"),
        resample_freq="1min",
    )
    cleaned_df = preprocessor.preprocess(str(raw_data_path))
    cleaned_df.to_csv(cleaned_path, index=False)

    print(f"[OK] Cleaned data saved: {cleaned_path}")
    print(f"[OK] Cleaned rows: {len(cleaned_df)}")

    print("\n" + "=" * 90)
    print("STEP 2/4 - FEATURE ENGINEERING")
    print("=" * 90)

    engineer = FeatureEngineer(
        belts_metadata_path=str(base_dir / "config" / "belts_metadata.json"),
        thresholds_path=str(base_dir / "config" / "thresholds.json"),
        target_mode="calendar_replace_date",
    )
    features_df = engineer.run(str(cleaned_path))
    features_df.to_csv(features_path, index=False)

    print(f"[OK] Features data saved: {features_path}")
    print(f"[OK] Feature rows: {len(features_df)}")
    print(f"[OK] Feature columns: {len(features_df.columns)}")

    _print_data_quality_and_degradation_report(cleaned_df, features_df)

    print("\n" + "=" * 90)
    print("STEP 3/4 - TRAINING ML MODELS")
    print("=" * 90)

    trainer = MLTrainer(str(base_dir / "config" / "model_config.json"))
    metrics = trainer.train_and_validate(str(features_path))

    print("\n" + "=" * 90)
    print("ML MODEL TRAINING METRICS")
    print("=" * 90)
    print(
        f"Samples: Total={metrics.get('n_total', 'N/A')}, "
        f"Train={metrics.get('n_train', 'N/A')}, "
        f"Test={metrics.get('n_test', 'N/A')}, "
        f"Features={metrics.get('n_features', 'N/A')}"
    )
    print(f"Split Date (UTC): {metrics.get('split_time_utc', 'N/A')}")
    print("-" * 90)
    print(f"{'Metric':<25} | {'Train R2':<12} | {'Train Adj-R2':<12} | {'Test R2':<12} | {'Test Adj-R2':<12}")
    print("-" * 90)

    h_train_r2 = metrics.get("health_train_r2", 0.0)
    h_train_adj_r2 = metrics.get("health_train_adj_r2", 0.0)
    h_test_r2 = metrics.get("health_test_r2", 0.0)
    h_test_adj_r2 = metrics.get("health_test_adj_r2", 0.0)

    r_train_r2 = metrics.get("rul_train_r2", 0.0)
    r_train_adj_r2 = metrics.get("rul_train_adj_r2", 0.0)
    r_test_r2 = metrics.get("rul_test_r2", 0.0)
    r_test_adj_r2 = metrics.get("rul_test_adj_r2", 0.0)

    print(f"{'Health Score':<25} | {h_train_r2:>12.4f} | {h_train_adj_r2:>12.4f} | {h_test_r2:>12.4f} | {h_test_adj_r2:>12.4f}")
    print(f"{'RUL (days)':<25} | {r_train_r2:>12.4f} | {r_train_adj_r2:>12.4f} | {r_test_r2:>12.4f} | {r_test_adj_r2:>12.4f}")
    print("-" * 90)

    print(f"Health Train MAE: {metrics.get('health_train_mae', 0.0):.4f}")
    print(f"Health Test  MAE: {metrics.get('health_test_mae', 0.0):.4f}")
    print(f"RUL Train MAE:    {metrics.get('rul_train_mae', 0.0):.4f}")
    print(f"RUL Test  MAE:    {metrics.get('rul_test_mae', 0.0):.4f}")

    if "health_cv_r2_mean" in metrics:
        print("\nTime-Series CV:")
        print(
            f"Health CV R2 Mean: {metrics.get('health_cv_r2_mean', 0.0):.4f} "
            f"(± {metrics.get('health_cv_r2_std', 0.0):.4f})"
        )
        print(
            f"RUL CV R2 Mean:    {metrics.get('rul_cv_r2_mean', 0.0):.4f} "
            f"(± {metrics.get('rul_cv_r2_std', 0.0):.4f})"
        )

    segment_order = ["normal", "warning", "critical"]
    if any(f"segment_{segment}_n" in metrics for segment in segment_order):
        print("\nSegment Metrics (Test Split):")
        for segment in segment_order:
            count = int(metrics.get(f"segment_{segment}_n", 0))
            if count <= 0:
                continue
            h_mae = metrics.get(f"segment_{segment}_health_mae", float("nan"))
            r_mae = metrics.get(f"segment_{segment}_rul_mae", float("nan"))
            h_r2 = metrics.get(f"segment_{segment}_health_r2", float("nan"))
            r_r2 = metrics.get(f"segment_{segment}_rul_r2", float("nan"))
            print(
                f"  - {segment.upper():<8} n={count:<6} "
                f"Health MAE={h_mae:.4f} R2={h_r2:.4f} | "
                f"RUL MAE={r_mae:.4f} R2={r_r2:.4f}"
            )

    print("\n" + "=" * 90)
    print("STEP 4/4 - TESTING SAVED MODELS")
    print("=" * 90)

    predictor = MLPredictor(
        models_dir=str(base_dir / "models" / "saved_models"),
        thresholds_config_path=str(base_dir / "config" / "thresholds.json"),
    )

    window_rows = max(int(prediction_window), 1)
    recent_df = features_df.tail(window_rows).copy()
    pred_df = predictor.predict(recent_df, force_ml_only=ml_only)

    if "@timestamp" in pred_df.columns:
        pred_df = pred_df.sort_values("@timestamp").reset_index(drop=True)
    latest = pred_df.iloc[-1]

    print("\n" + "+" + "-" * 62 + "+")
    print("|" + "BELT HEALTH & RUL - ML PREDICTION".center(62) + "|")
    print("+" + "-" * 62 + "+")
    print(f"| {'Item':<15} | {'Value':<12} | {'Description':<27} |")
    print("+" + "-" * 62 + "+")
    if ml_only:
        health_value = float(latest.get("ml_health", 0.0))
        rul_value = float(latest.get("ml_rul_days", 0.0))
        health_desc = "Pure ML belt health %"
        rul_desc = "Pure ML remaining life"
    else:
        health_value = float(latest.get("final_health", 0.0))
        rul_value = float(latest.get("final_rul_days", 0.0))
        health_desc = "Predicted belt health %"
        rul_desc = "Remaining Useful Life"

    print(f"| {'Health Score':<15} | {health_value:>12.1f} | {health_desc:<27} |")
    print(f"| {'RUL (days)':<15} | {rul_value:>12.1f} | {rul_desc:<27} |")
    print(f"| {'RUL (months)':<15} | {rul_value / 30.4:>12.1f} | {'Approximate months rem.':<27} |")
    print("+" + "-" * 62 + "+")

    print(f"Prediction Status: {latest.get('prediction_status', 'unknown')}")
    print(f"Prediction Error: {latest.get('prediction_error', '')}")
 
    if latest.get("prediction_status") not in {"ok", "ok_ml_only"}:
        print(f"WARNING: Fallback values (NaN) returned due to: {latest.get('prediction_error')}")

    print("\n[OK] ML standalone pipeline completed successfully.")

    # --- Save Results to results/ directory ---
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Metrics
    metrics_path = results_dir / "run_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[OK] Metrics saved to: {metrics_path}")

    # 2. Save Final Prediction Summary
    summary_path = results_dir / "prediction_summary.json"
    summary_health = float(latest.get("ml_health", 0.0)) if ml_only else float(latest.get("final_health", 0.0))
    summary_rul = float(latest.get("ml_rul_days", 0.0)) if ml_only else float(latest.get("final_rul_days", 0.0))

    summary = {
        "timestamp_utc": str(latest.get("@timestamp")),
        "health_score": summary_health,
        "rul_days": summary_rul,
        "status": str(latest.get("prediction_status", "unknown")),
        "confidence": str(latest.get("confidence_level", "LOW")),
        "prediction_mode": "ml_only" if ml_only else "hybrid",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"[OK] Prediction summary saved to: {summary_path}")

    # 3. Save Full Prediction DF
    preds_csv_path = results_dir / "predictions.csv"
    pred_df.to_csv(preds_csv_path, index=False)
    print(f"[OK] Full predictions saved to: {preds_csv_path}")


if __name__ == "__main__":
    cli_args = _parse_args()
    run_ml_pipeline(
        raw_data_path_str=cli_args.raw_data,
        ml_only=cli_args.ml_only,
        prediction_window=cli_args.prediction_window,
    )