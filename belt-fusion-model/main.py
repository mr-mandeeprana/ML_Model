"""
Unified Entry Point for Belt Health & RUL ML Model
Runs the ML pipeline and reports predicted belt health and remaining useful life.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import warnings

import pandas as pd

from ml_model.data_preprocessing import DataPreprocessor
from ml_model.feature_engineering import FeatureEngineer
from ml_model.predictor import MLPredictor

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ML-only belt health and RUL prediction.")
    parser.add_argument(
        "--raw-data",
        default="data/bucket_elevator_synthetic_failure_1min_4months.csv",
        help="Path to raw long-format sensor CSV, relative to project root or absolute.",
    )
    parser.add_argument(
        "--features",
        default="data/processed/features.csv",
        help="Path to engineered features CSV, relative to project root or absolute.",
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Regenerate cleaned/features from raw CSV even if cached features exist.",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Use strict ML outputs only (no calendar fallback or blending).",
    )
    return parser.parse_args()


def _resolve_path(base_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return base_dir / path


def _load_thresholds(base_dir: Path) -> dict:
    with open(base_dir / "config" / "thresholds.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _assess_data_quality(window_df: pd.DataFrame, required_features: list[str], thresholds: dict) -> tuple[bool, dict]:
    quality_cfg = thresholds.get("data_quality", {})
    min_rows_required = int(quality_cfg.get("min_rows_required", 100))
    max_missing_ratio = float(quality_cfg.get("max_missing_ratio", 0.3))

    feature_cols = [c for c in required_features if c in window_df.columns]
    feature_frame = window_df[feature_cols].copy() if feature_cols else pd.DataFrame(index=window_df.index)

    for col in feature_cols:
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")

    if feature_frame.empty:
        missing_ratio = 1.0
    else:
        missing_ratio = float(feature_frame.isna().sum().sum() / max(feature_frame.size, 1))

    quality_ok = len(window_df) >= min_rows_required and missing_ratio <= max_missing_ratio
    details = {
        "row_count": len(window_df),
        "min_rows_required": min_rows_required,
        "required_feature_count": len(feature_cols),
        "missing_ratio": missing_ratio,
        "max_missing_ratio": max_missing_ratio,
    }
    return quality_ok, details


def _build_risk_reasons(latest: pd.Series, prediction_status: str, quality_ok: bool) -> list[str]:
    reasons: list[str] = []

    if not quality_ok:
        reasons.append("data_quality_gate_failed")
    if prediction_status not in {"ok", "ok_ml_only", "safe_mode_calendar"}:
        reasons.append(f"prediction_status={prediction_status}")
    if int(latest.get("critical_count", 0)) > 0:
        reasons.append(f"critical_count={int(latest.get('critical_count', 0))}")
    if int(latest.get("warning_count", 0)) > 0:
        reasons.append(f"warning_count={int(latest.get('warning_count', 0))}")
    if int(latest.get("condition_state_code", 0)) == 2:
        reasons.append("condition_state=CRITICAL")
    elif int(latest.get("condition_state_code", 0)) == 1:
        reasons.append("condition_state=WARNING")
    if int(latest.get("degradation_rising_flag", 0)) > 0:
        reasons.append("degradation_rising")
    if float(latest.get("degradation_index", 0.0)) >= 0.8:
        reasons.append(f"degradation_index={float(latest.get('degradation_index', 0.0)):.3f}")

    return reasons


def main() -> None:
    print("=" * 70)
    print("     BELT HEALTH & RUL INITIALIZATION (ML ONLY)")
    print("=" * 70)

    args = _parse_args()

    base_dir = Path(__file__).parent
    raw_data_path = _resolve_path(base_dir, args.raw_data)
    cleaned_path = base_dir / "data" / "processed" / "cleaned_sensor_data.csv"
    features_path = _resolve_path(base_dir, args.features)

    features_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load or generate features
    if features_path.exists() and not args.force_preprocess:
        print("Loading cached feature data...")
        df = pd.read_csv(features_path)
        if "@timestamp" in df.columns:
            df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["@timestamp"]).reset_index(drop=True)
    else:
        print("Preprocessed features not found. Running pipeline first...")

        if not raw_data_path.exists():
            print(f"ERROR: Raw data missing at {raw_data_path}")
            return

        preprocessor = DataPreprocessor(
            config_path=str(base_dir / "config" / "thresholds.json"),
            resample_freq="1min",
        )
        engineer = FeatureEngineer(
            belts_metadata_path=str(base_dir / "config" / "belts_metadata.json"),
            thresholds_path=str(base_dir / "config" / "thresholds.json"),
            target_mode="calendar_replace_date",
        )

        print("Preprocessing data...")
        cleaned_df = preprocessor.preprocess(str(raw_data_path))
        cleaned_df.to_csv(cleaned_path, index=False)

        print("Feature engineering...")
        df = engineer.run(str(cleaned_path))
        df.to_csv(features_path, index=False)

    if df.empty:
        print("ERROR: No feature data available.")
        return

    print(f"Data ready: {len(df)} records.\n")

    # 2. Initialize predictor
    print("Initializing ML Predictor...")
    ml_predictor = MLPredictor(
        models_dir=str(base_dir / "models" / "saved_models"),
        thresholds_config_path=str(base_dir / "config" / "thresholds.json"),
    )

    if not ml_predictor.is_ready():
        print("ERROR: ML models are not ready. Train models first using ml_model/run_standalone.py")
        return

    print("Running predictions on latest records...")

    thresholds = _load_thresholds(base_dir)
    latest_rows = df.tail(100).copy()
    pred_df = ml_predictor.predict(latest_rows, force_ml_only=args.ml_only)

    if "@timestamp" in pred_df.columns:
        pred_df = pred_df.sort_values("@timestamp").reset_index(drop=True)
    latest = pred_df.iloc[-1]

    # Extraction
    ml_health = pd.to_numeric(latest.get("ml_health"), errors="coerce")
    ml_rul = pd.to_numeric(latest.get("ml_rul_days"), errors="coerce")
    calendar_rul = pd.to_numeric(latest.get("calendar_rul_days"), errors="coerce")
    final_health = pd.to_numeric(latest.get("final_health"), errors="coerce")
    final_rul = pd.to_numeric(latest.get("final_rul_days"), errors="coerce")
    confidence_level = str(latest.get("confidence_level", "LOW"))
    prediction_status = str(latest.get("prediction_status", "unknown"))
    is_safe_mode = prediction_status == "safe_mode_calendar"

    if pd.isna(final_health) or pd.isna(final_rul):
        if prediction_status == "ok":
            prediction_status = "fallback_nan_prediction"

    quality_ok, quality_details = _assess_data_quality(latest_rows, ml_predictor.features, thresholds)

    health_thresholds = thresholds.get("health_score_thresholds", {})
    rul_thresholds = thresholds.get("rul_thresholds", {})

    health_critical = float(health_thresholds.get("critical", 50.0))
    health_warning = float(health_thresholds.get("warning", 65.0))
    rul_critical = float(rul_thresholds.get("critical_days", 90.0))
    rul_warning = float(rul_thresholds.get("warning_days", 180.0))

    warning_count = int(latest.get("warning_count", 0))
    critical_count = int(latest.get("critical_count", 0))
    condition_state_code = int(latest.get("condition_state_code", 0))
    degradation_index = float(latest.get("degradation_index", 0.0))
    degradation_rising_flag = int(latest.get("degradation_rising_flag", 0))
    elong_trend_6h = float(latest.get("elong_trend_6h", 0.0))

    sensor_critical = (
        (critical_count >= 2)
        and (
            degradation_index >= 0.80
            or degradation_rising_flag > 0
            or elong_trend_6h > 0.01
            or condition_state_code == 2
        )
    )

    sensor_warning = (
        (warning_count >= 1)
        or (condition_state_code == 1)
        or (degradation_index >= 0.45)
    )

    safe_health = 100.0 if pd.isna(final_health) else float(final_health)
    safe_rul = 2190.0 if pd.isna(final_rul) else float(final_rul)

    model_critical = (
        safe_rul <= rul_critical
        and (degradation_index >= 0.80 or critical_count >= 2)
    )
    model_warning = (
        safe_rul <= rul_warning
        or safe_health <= health_warning
        or sensor_warning
    )

    if not quality_ok or confidence_level == "LOW":
        risk = "LOW_CONFIDENCE"
    elif sensor_critical or model_critical:
        risk = "CRITICAL"
    elif model_warning:
        risk = "WARNING"
    else:
        risk = "NORMAL"

    risk_reasons = _build_risk_reasons(latest, prediction_status, quality_ok)

    # 3. Print final result table
    display_health = "NaN" if pd.isna(final_health) else f"{final_health:>12.1f}"
    display_ml_rul = "NaN" if pd.isna(ml_rul) else f"{ml_rul:>12.1f}"
    display_final_rul = "NaN" if pd.isna(final_rul) else f"{final_rul:>12.1f}"
    display_months = "NaN" if pd.isna(final_rul) else f"{final_rul / 30.4:>12.1f}"
    raw_rul_label = "Fallback RUL (days)" if is_safe_mode else "ML RUL (days)"
    raw_rul_desc = "Calendar fallback remaining life" if is_safe_mode else "Raw model remaining life"

    if args.ml_only:
        display_health = "NaN" if pd.isna(ml_health) else f"{ml_health:>12.1f}"
        display_final_rul = "NaN" if pd.isna(ml_rul) else f"{ml_rul:>12.1f}"
        display_months = "NaN" if pd.isna(ml_rul) else f"{ml_rul / 30.4:>12.1f}"
        raw_rul_label = "ML RUL (days)"
        raw_rul_desc = "Pure ML remaining life"

    print("\n" + "+" + "-" * 78 + "+")
    print("|" + "BELT HEALTH & RUL - FINAL ML OUTPUT".center(78) + "|")
    print("+" + "-" * 78 + "+")
    print(f"| {'Item':<20} | {'Value':<12} | {'Description':<39} |")
    print("+" + "-" * 78 + "+")
    health_desc = "Pure ML predicted belt health %" if args.ml_only else "Final predicted belt health %"
    print(f"| {'Health Score':<20} | {display_health} | {health_desc:<39} |")
    print(f"| {raw_rul_label:<20} | {display_ml_rul} | {raw_rul_desc:<39} |")
    if not args.ml_only:
        display_calendar_rul = "NaN" if pd.isna(calendar_rul) else f"{calendar_rul:>12.1f}"
        print(f"| {'Calendar RUL (days)':<20} | {display_calendar_rul} | {'Lifecycle-based remaining life':<39} |")
        print(f"| {'Final RUL (days)':<20} | {display_final_rul} | {'Hybrid remaining useful life':<39} |")
    else:
        print(f"| {'Final RUL (days)':<20} | {display_final_rul} | {'Pure ML remaining useful life':<39} |")
    print(f"| {'RUL (months)':<20} | {display_months} | {'Approximate months remaining':<39} |")
    print("+" + "-" * 78 + "+")

    print(f"\nRisk Level: {risk}")
    print(f"Confidence Level: {confidence_level}")
    if args.ml_only:
        print("Prediction Mode: ML_ONLY")
    else:
        print(f"Prediction Mode: {'SAFE_MODE_FALLBACK' if is_safe_mode else 'ML_MODEL'}")
    print(f"Prediction Status: {latest.get('prediction_status', 'unknown')}")
    print(f"Prediction Error: {latest.get('prediction_error', '')}")

    if latest.get("prediction_status") == "safe_mode_calendar":
        print("INFO: Safe-mode calendar fallback used for raw ML fields.")
    elif latest.get("prediction_status") not in {"ok", "ok_ml_only"}:
        fallback_reason = latest.get("prediction_error") or latest.get("prediction_status")
        print(f"WARNING: Fallback prediction used due to: {fallback_reason}")
    print(f"Warning Flags: {int(latest.get('warning_count', 0))}")
    print(f"Condition State: {latest.get('condition_state', 'UNKNOWN')}")
    print(
        f"Data Quality: rows={quality_details['row_count']}, missing_ratio={quality_details['missing_ratio']:.3f}"
    )

    if risk_reasons:
        print("Risk Reasons: " + ", ".join(risk_reasons[:6]))

    if "@timestamp" in latest.index:
        print(f"Prediction Timestamp: {latest['@timestamp']}")

    # 4. Save results to same file
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = results_dir / "prediction_summary.json"
    summary = {
        "timestamp_utc": str(latest.get("@timestamp")),
        "health_score": float(final_health),
        "rul_days": float(final_rul),
        "status": str(prediction_status),
        "confidence": str(confidence_level),
        "prediction_mode": "ml_only" if args.ml_only else "hybrid",
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Save full predictions for these records
    preds_csv_path = results_dir / "predictions.csv"
    pred_df.to_csv(preds_csv_path, index=False)

    print(f"\n[OK] Results saved to {results_dir}")
    print("[OK] ML prediction completed successfully.")


if __name__ == "__main__":
    main()