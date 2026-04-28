"""
ML Predictor
Loads trained ML models to predict Health Score and RUL (days).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MLPredictor:
    def __init__(
        self,
        models_dir: str = "models/saved_models",
        thresholds_config_path: str = "config/thresholds.json",
    ) -> None:
        self.models_dir = Path(models_dir)
        self.thresholds_config_path = Path(thresholds_config_path)

        self.baseline_rul_days = self._load_baseline_rul_days()

        self.health_artifact = self._load_model("ml_health_model.pkl")
        self.rul_artifact = self._load_model("ml_rul_model.pkl")

        self.health_model = None
        self.rul_model = None
        self.scaler = None
        self.features: List[str] = []

        self._initialize_artifacts()

    def _load_baseline_rul_days(self) -> float:
        if self.thresholds_config_path.exists():
            try:
                with open(self.thresholds_config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return float(cfg.get("baseline_values", {}).get("baseline_rul_days", 2190.0))
            except Exception as exc:
                logger.warning(
                    "Failed to load baseline RUL from %s: %s",
                    self.thresholds_config_path,
                    exc,
                )
        return 2190.0

    def _load_model(self, filename: str) -> Optional[Dict]:
        path = self.models_dir / filename
        if not path.exists():
            logger.warning("Model artifact not found: %s", path)
            return None

        try:
            artifact = joblib.load(path)
            logger.info("Loaded model artifact: %s", path)
            return artifact
        except Exception as exc:
            logger.error("Failed to load artifact %s: %s", path, exc)
            return None

    def _initialize_artifacts(self) -> None:
        if self.health_artifact is None or self.rul_artifact is None:
            logger.warning("One or both model artifacts are missing.")
            return

        self.health_model = self.health_artifact.get("model")
        self.rul_model = self.rul_artifact.get("model")

        self.scaler = self.health_artifact.get("scaler")
        self.features = list(self.health_artifact.get("features", []))

        rul_features = list(self.rul_artifact.get("features", []))
        if self.features != rul_features:
            logger.warning("Health and RUL artifacts have different feature lists. Using health artifact features.")

        if self.health_model is None or self.rul_model is None:
            logger.warning("Loaded artifacts are missing model objects.")

        if not self.features:
            logger.warning("No feature list found inside artifact.")

        if self.scaler is None:
            logger.warning("No scaler found in artifact. Predictor will fall back to raw feature values.")

    def is_ready(self) -> bool:
        return (
            self.health_model is not None
            and self.rul_model is not None
            and len(self.features) > 0
        )

    def _build_feature_frame(self, df_features: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.features if col not in df_features.columns]
        if missing:
            raise ValueError(
                "Missing required inference features: "
                + ", ".join(missing[:15])
                + (" ..." if len(missing) > 15 else "")
            )

        X_df = pd.DataFrame(index=df_features.index)

        for col in self.features:
            X_df[col] = pd.to_numeric(df_features[col], errors="coerce")

        null_cols = [col for col in self.features if X_df[col].isna().any()]
        if null_cols:
            raise ValueError(
                "Inference features contain NaN after numeric conversion: "
                + ", ".join(null_cols[:15])
                + (" ..." if len(null_cols) > 15 else "")
            )

        return X_df

    def predict(self, df_features: pd.DataFrame, force_ml_only: bool = False) -> pd.DataFrame:
        df = df_features.copy()

        df["ml_health"] = np.nan
        df["ml_rul_days"] = np.nan
        df["final_rul_days"] = np.nan
        df["final_health"] = np.nan
        df["confidence_level"] = "LOW"
        df["prediction_status"] = "fallback_uninitialized"
        df["prediction_error"] = "predictor_not_ready"


        # In strict ML-only mode, never fall back to calendar when models are unavailable.
        if not self.is_ready() and force_ml_only:
            logger.error("ML-ONLY MODE requested but predictor artifacts are not ready")
            df["prediction_status"] = "ml_only_model_unavailable"
            df["prediction_error"] = "ml_models_not_ready"
            return df

        # SAFE MODE when models are unavailable.
        if not self.is_ready():
            logger.info("SAFE MODE: Using calendar + degradation model (no ML)")

            baseline = self.baseline_rul_days

            # Ensure calendar RUL exists
            if "calendar_rul_days" in df.columns:
                cal_rul = pd.to_numeric(df["calendar_rul_days"], errors="coerce")
            else:
                cal_rul = pd.Series(index=df.index, dtype=float)

            # HARD fallback if NaN
            if cal_rul.isna().all():
                logger.warning("calendar_rul_days missing → using baseline fallback")
                cal_rul = pd.Series(baseline, index=df.index)

            # Fill partial NaNs
            cal_rul = cal_rul.ffill().fillna(baseline)

            # Keep the raw ML-facing columns numeric in safe mode so result
            # tables and exported CSVs do not surface NaN values.
            df["ml_rul_days"] = cal_rul.clip(lower=0.0, upper=baseline)
            df["ml_health"] = np.clip(100.0 * (df["ml_rul_days"] / baseline), 0.0, 100.0)

            # Degradation adjustment (important)
            degradation_index = pd.to_numeric(df.get("degradation_index", 0.0), errors="coerce").fillna(0.0)
            warning_count = pd.to_numeric(df.get("warning_count", 0), errors="coerce").fillna(0.0)

            penalty = 1.0 - np.clip(0.15 * degradation_index + 0.05 * warning_count, 0, 0.5)

            final_rul = cal_rul * penalty

            df["final_rul_days"] = final_rul.clip(lower=0.0, upper=baseline)
            df["final_health"] = 100.0 * (df["final_rul_days"] / baseline)

            # Final safety fill (Fix 3)
            df["final_rul_days"] = df["final_rul_days"].fillna(baseline)
            df["final_health"] = df["final_health"].fillna(100.0)

            df["confidence_level"] = "MEDIUM"
            df["prediction_status"] = "safe_mode_calendar"
            df["prediction_error"] = ""

            return df


        try:
            X_df = self._build_feature_frame(df)
        except Exception as exc:
            logger.error("Feature preparation failed: %s", exc)
            df["prediction_status"] = "fallback_feature_prep_failed"
            df["prediction_error"] = str(exc)
            return df

        X = X_df.values

        if self.scaler is not None:
            try:
                X = self.scaler.transform(X)
            except Exception as exc:
                logger.error("Scaler transform failed: %s", exc)
                df["prediction_status"] = "fallback_scaler_failed"
                df["prediction_error"] = str(exc)
                return df

        try:
            health_preds = self.health_model.predict(X)
            rul_preds_raw = self.rul_model.predict(X)

            pred_health = np.clip(health_preds, 0.0, 100.0)

            target_transform = self.rul_artifact.get("target_transform", "none")
            if target_transform == "log1p":
                pred_rul = np.expm1(rul_preds_raw)
            else:
                pred_rul = rul_preds_raw

            pred_rul = np.clip(pred_rul, 0.0, self.baseline_rul_days)

            df["ml_health_raw"] = pred_health
            df["ml_rul_raw"] = pred_rul
            df["ml_health"] = pd.Series(pred_health, index=df.index).rolling(15, min_periods=1).mean()
            df["ml_rul_days"] = pd.Series(pred_rul, index=df.index).rolling(15, min_periods=1).mean()

            # Preserve sharp failures
            df["ml_health"] = np.minimum(df["ml_health"], df["ml_health_raw"])
            df["ml_rul_days"] = np.minimum(df["ml_rul_days"], df["ml_rul_raw"])

            df["final_health"] = df["ml_health"]
            df["final_rul_days"] = df["ml_rul_days"]

            if not force_ml_only:
                degradation_index = pd.to_numeric(df.get("degradation_index", 0.0), errors="coerce").fillna(0.0)
                warning_count = pd.to_numeric(df.get("warning_count", 0), errors="coerce").fillna(0.0)
                critical_count = pd.to_numeric(df.get("critical_count", 0), errors="coerce").fillna(0.0)

                penalty = np.clip(
                    1.0 - (0.08 * degradation_index + 0.03 * warning_count + 0.08 * critical_count),
                    0.65,
                    1.0,
                )

                df["final_rul_days"] = df["ml_rul_days"] * penalty
                rul_health = 100.0 * df["final_rul_days"] / self.baseline_rul_days
                df["final_health"] = (0.7 * df["ml_health"] + 0.3 * rul_health)

            df["final_health"] = df["final_health"].clip(0.0, 100.0)
            df["final_rul_days"] = df["final_rul_days"].clip(0.0, self.baseline_rul_days)

            # Fix 2: Confidence logic
            feature_std = np.std(X, axis=1)
            df["confidence_level"] = "HIGH"
            df.loc[feature_std < 0.01, "confidence_level"] = "MEDIUM"
            df.loc[(df["ml_rul_days"] < 50) | (df["ml_health"] < 40), "confidence_level"] = "LOW"

            # Fix 7: Failure awareness
            df["failure_risk_score"] = (
                0.5 * df.get("critical_count", 0) +
                0.3 * df.get("warning_count", 0) +
                0.4 * df.get("degradation_index", 0)
            ).clip(0, 5)

            # Optional: Anomaly score
            df["anomaly_score"] = (
                df.get("current_volatility_6h", 0) +
                df.get("temp_variability_12h", 0)
            )
            df["anomaly_score"] = df["anomaly_score"] / (df["anomaly_score"].max() + 1e-6)

            # Early failure trigger
            df["early_failure_flag"] = (
                (df.get("degradation_index", 0) > 0.6) &
                (df.get("elong_trend_6h", 0) > 0.01)
            ).astype(int)
            df["prediction_status"] = "ok_ml_only" if force_ml_only else "ok"
            df["prediction_error"] = ""

            return df

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            df["prediction_status"] = "fallback_prediction_failed"
            df["prediction_error"] = str(exc)
            return df