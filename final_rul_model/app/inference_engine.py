"""
Inference Engine
Loads trained ML artifacts and performs real-time prediction.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class InferenceEngine:
    def __init__(
        self,
        model_dir: str = "models/saved_models",
        thresholds_path: str = "config/thresholds.json",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.thresholds_path = Path(thresholds_path)
        self.alt_model_dirs = [
            Path(model_dir),
            Path("models/saved_models"),
            Path("/app/models/saved_models"),
            Path("model"),
            Path("/app/model"),
        ]

        self.baseline_rul_days = self._load_baseline_rul_days()

        self.health_artifact = self._load_artifact_any([
            "belt_rul_model_health.pkl",
            "ml_health_model.pkl",
        ])
        self.rul_artifact = self._load_artifact_any([
            "belt_rul_model_rul.pkl",
            "ml_rul_model.pkl",
        ])

        self.health_model = None
        self.rul_model = None
        self.scaler = None
        self.features: List[str] = []
        self._warned_missing_features = set()

        self._initialize()

    def _load_baseline_rul_days(self) -> float:
        env_thresholds = os.getenv("THRESHOLDS_PATH")
        if env_thresholds:
            self.thresholds_path = Path(env_thresholds)

        if self.thresholds_path.exists():
            try:
                with open(self.thresholds_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                return float(cfg.get("baseline_values", {}).get("baseline_rul_days", 2190.0))
            except Exception as exc:
                logger.warning("Could not load baseline RUL from %s: %s", self.thresholds_path, exc)

        mounted_thresholds = Path("/config/thresholds.json")
        if mounted_thresholds.exists():
            try:
                with open(mounted_thresholds, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                self.thresholds_path = mounted_thresholds
                return float(cfg.get("baseline_values", {}).get("baseline_rul_days", 2190.0))
            except Exception as exc:
                logger.warning("Could not load baseline RUL from %s: %s", mounted_thresholds, exc)

        return 2190.0

    def _load_artifact(self, filename: str) -> Optional[Dict]:
        path = self.model_dir / filename
        if not path.exists():
            logger.error("Model artifact not found: %s", path)
            return None

        try:
            artifact = joblib.load(path)
            logger.info("Loaded artifact: %s", path)
            return artifact
        except Exception as exc:
            logger.error("Failed to load artifact %s: %s", path, exc)
            return None

    def _load_artifact_any(self, filenames: List[str]) -> Optional[Dict]:
        for model_dir in self.alt_model_dirs:
            self.model_dir = model_dir
            for filename in filenames:
                artifact = self._load_artifact(filename)
                if artifact is not None:
                    return artifact
        logger.error("None of artifact candidates found. Tried files: %s", filenames)
        return None

    def _initialize(self) -> None:
        if self.health_artifact is None or self.rul_artifact is None:
            logger.warning("Health or RUL artifact missing.")
            return

        self.health_model = self.health_artifact.get("model")
        self.rul_model = self.rul_artifact.get("model")
        self.scaler = self.health_artifact.get("scaler")
        self.features = list(self.health_artifact.get("features", []))

        rul_features = list(self.rul_artifact.get("features", []))
        if self.features != rul_features:
            logger.warning("Health and RUL feature lists differ. Using health artifact feature list.")

        if not self.features:
            logger.warning("No feature schema found in model artifact.")
        if self.scaler is None:
            logger.warning("No scaler found in model artifact. Raw feature values will be used.")

    def is_ready(self) -> bool:
        return (
            self.health_model is not None
            and self.rul_model is not None
            and len(self.features) > 0
        )

    def _build_feature_frame(self, features: Dict[str, float]) -> pd.DataFrame:
        row = {}
        for col in self.features:
            if col not in features and col not in self._warned_missing_features:
                logger.warning("Missing required feature: %s", col)
                self._warned_missing_features.add(col)
            
            value = features.get(col, 0.0)
            try:
                row[col] = float(value)
            except (TypeError, ValueError):
                row[col] = 0.0
        
        return pd.DataFrame([row], columns=self.features)

    def predict(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Performs inference with SAFE MODE fallback and confidence scoring.
        """
        # SAFE MODE when models are unavailable
        if not self.is_ready():
            baseline = self.baseline_rul_days
            cal_rul = float(features.get("calendar_rul_days", baseline))
            
            # Simple degradation adjustment for safe mode
            deg_idx = float(features.get("degradation_index", 0.0))
            warn_cnt = float(features.get("warning_count", 0))
            penalty = 1.0 - np.clip(0.15 * deg_idx + 0.05 * warn_cnt, 0, 0.5)
            
            final_rul = float(np.clip(cal_rul * penalty, 0.0, baseline))
            final_health = 100.0 * (final_rul / baseline)
            
            return {
                "ml_health": final_health,
                "ml_rul_days": final_rul,
                "confidence_level": "MEDIUM",
                "prediction_status": "safe_mode_calendar",
                "failure_risk_score": float(np.clip(deg_idx * 2.0, 0, 5))
            }

        try:
            X_df = self._build_feature_frame(features)
            X = X_df.values

            if self.scaler is not None:
                X = self.scaler.transform(X)

            # Raw predictions
            health_pred_raw = float(self.health_model.predict(X)[0])
            rul_pred_raw = float(self.rul_model.predict(X)[0])

            # Handle log transformation if used during training
            target_transform = self.rul_artifact.get("target_transform", "none")
            if target_transform == "log1p":
                rul_pred = float(np.expm1(rul_pred_raw))
            else:
                rul_pred = rul_pred_raw

            # Post-processing / Smoothing (Single point version)
            ml_health = float(np.clip(health_pred_raw, 0.0, 100.0))
            ml_rul = float(np.clip(rul_pred, 0.0, self.baseline_rul_days))

            # Penalty adjustment (Logic from source)
            deg_idx = float(features.get("degradation_index", 0.0))
            warn_cnt = float(features.get("warning_count", 0))
            crit_cnt = float(features.get("critical_count", 0))
            
            penalty = float(np.clip(
                1.0 - (0.08 * deg_idx + 0.03 * warn_cnt + 0.08 * crit_cnt),
                0.65,
                1.0,
            ))

            final_rul = ml_rul * penalty
            rul_health = 100.0 * final_rul / self.baseline_rul_days
            final_health = float(np.clip(0.7 * ml_health + 0.3 * rul_health, 0.0, 100.0))
            final_rul = float(np.clip(final_rul, 0.0, self.baseline_rul_days))

            # Confidence Level
            # Using feature variance across input as a proxy for OOD if scaler is used
            conf_level = "HIGH"
            if np.std(X) < 0.01:
                conf_level = "MEDIUM"
            if final_rul < 50 or final_health < 40:
                conf_level = "LOW"

            return {
                "ml_health": final_health,
                "ml_rul_days": final_rul,
                "confidence_level": conf_level,
                "prediction_status": "ok",
                "failure_risk_score": float(np.clip(0.5 * crit_cnt + 0.3 * warn_cnt + 0.4 * deg_idx, 0, 5)),
                "early_failure_flag": int(deg_idx > 0.6 and float(features.get("elong_trend_6h", 0)) > 0.01)
            }

        except Exception as exc:
            logger.error("Prediction failed: %s", exc)
            return {
                "ml_health": 0.0,
                "ml_rul_days": 0.0,
                "prediction_status": "failed",
                "prediction_error": str(exc)
            }