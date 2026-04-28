"""
Feature Engineering for Belt Health & RUL Prediction
Builds runtime-compatible features from preprocessed sensor data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds training features from long-format sensor data.

    Input expected:
    - sensorid
    - @timestamp
    - avg_value

    Output:
    - wide dataframe with engineered features
    - target_rul_days
    - target_health (derived from RUL for compatibility)
    """

    CORE_SENSORS = [
        "temperature_boot_material/temperature",
        "ultrasonic_boot/elongation",
        "current_transducer_head/current",
    ]

    def __init__(
        self,
        belts_metadata_path: str = "config/belts_metadata.json",
        thresholds_path: str = "config/thresholds.json",
        target_mode: str = "calendar_replace_date",
    ) -> None:
        self.belts_metadata_path = Path(belts_metadata_path)
        self.thresholds_path = Path(thresholds_path)
        self.target_mode = target_mode

        self.metadata = self._load_json(self.belts_metadata_path)
        self.thresholds = self._load_json(self.thresholds_path)

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            logger.warning("Config file not found at %s. Using empty dict.", path)
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", path, exc)
            return {}

    def pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True)
        df = df[df["sensorid"].isin(self.CORE_SENSORS)].copy()

        wide_df = df.pivot_table(
            index="@timestamp",
            columns="sensorid",
            values="avg_value",
            aggfunc="mean",
        ).sort_index()

        expected_cols = [f"{sensor}_avg" for sensor in self.CORE_SENSORS]
        wide_df.columns = [f"{col}_avg" for col in wide_df.columns]

        for col in expected_cols:
            if col not in wide_df.columns:
                wide_df[col] = np.nan

        wide_df = wide_df[expected_cols].sort_index()
        wide_df = wide_df.interpolate(limit_direction="both")
        wide_df = wide_df.ffill().bfill()

        return wide_df.reset_index()

    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().set_index("@timestamp").sort_index()

        windows = {
            "1h": 60,
            "6h": 360,
            "12h": 720,
            "24h": 1440,
        }

        base_cols = [
            "temperature_boot_material/temperature_avg",
            "ultrasonic_boot/elongation_avg",
            "current_transducer_head/current_avg",
        ]

        for col in base_cols:
            if col not in df.columns:
                continue

            base = col.replace("_avg", "")

            for label, window in windows.items():
                df[f"{base}_{label}_mean"] = df[col].rolling(window, min_periods=1).mean()
                df[f"{base}_{label}_std"] = df[col].rolling(window, min_periods=2).std().fillna(0.0)
                df[f"{base}_{label}_min"] = df[col].rolling(window, min_periods=1).min()
                df[f"{base}_{label}_max"] = df[col].rolling(window, min_periods=1).max()

        return df.reset_index()

    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        current_col = "current_transducer_head/current_avg"
        if current_col not in df.columns:
            return df

        idle_threshold = float(
            self.thresholds.get("idle_detection", {}).get("current_threshold", 4.5)
        )
        high_load_threshold = float(
            self.thresholds.get("operational_states", {}).get("high_load_threshold", 62.0)
        )

        df["is_idle"] = (df[current_col] < idle_threshold).astype(int)
        df["high_load"] = (df[current_col] >= high_load_threshold).astype(int)

        active_flag = (~df["is_idle"].astype(bool)).astype(int)
        df["utilization_1h"] = active_flag.rolling(window=60, min_periods=1).mean()
        df["utilization_6h"] = active_flag.rolling(window=360, min_periods=1).mean()
        df["high_load_share_6h"] = df["high_load"].rolling(window=360, min_periods=1).mean()

        df["current_percentile"] = (
            df[current_col]
            .rolling(window=720, min_periods=10)
            .apply(
                lambda x: ((x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6)) * 100.0,
                raw=False,
            )
            .fillna(0.0)
        )

        df["current_volatility_6h"] = (
            df[current_col].rolling(window=360, min_periods=10).std().fillna(0.0)
        )

        return df

    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        def calc(values: np.ndarray) -> float:
            if len(values) < 2:
                return 0.0
            x = np.arange(len(values), dtype=float)
            y = values.astype(float)
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        return series.rolling(window=window, min_periods=max(5, min(window, 10))).apply(
            calc,
            raw=True,
        ).fillna(0.0)

    def create_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        elong_col = "ultrasonic_boot/elongation_avg"
        temp_col = "temperature_boot_material/temperature_avg"

        baseline_elong = float(
            self.thresholds.get("baseline_values", {}).get("baseline_elongation", 280.0)
        )
        baseline_temp = float(
            self.thresholds.get("baseline_values", {}).get("baseline_temp_celsius", 60.0)
        )
        critical_elong = float(
            self.thresholds.get("sensor_thresholds", {})
            .get("ultrasonic_boot/elongation", {})
            .get("critical", 310.0)
        )

        if elong_col in df.columns:
            df["elong_rate_12h"] = df[elong_col].diff(720).fillna(0.0)
            df["elong_delta"] = df[elong_col] - baseline_elong
            df["elong_delta_rate"] = df["elong_delta"].diff(60).fillna(0.0)

            df["elong_trend_1h"] = self._rolling_slope(df[elong_col], 60)
            df["elong_trend_6h"] = self._rolling_slope(df[elong_col], 360)
            df["elong_trend_24h"] = self._rolling_slope(df[elong_col], 1440)

            df["elong_micro_trend"] = df[elong_col].diff(5).fillna(0.0)
            df["elong_accel_6h"] = (df["elong_trend_1h"] - df["elong_trend_6h"]).fillna(0.0)
            df["elong_volatility_6h"] = (
                df[elong_col].rolling(window=360, min_periods=10).std().fillna(0.0)
            )

            elong_span = max(critical_elong - baseline_elong, 1.0)
            delta_norm = np.clip(df["elong_delta"] / elong_span, 0.0, 2.0)
            trend_norm = np.clip(df["elong_trend_6h"] / 0.03, 0.0, 2.0)
            vol_norm = np.clip(df["elong_volatility_6h"] / 6.0, 0.0, 2.0)

            df["degradation_index"] = np.clip(
                0.55 * delta_norm + 0.30 * trend_norm + 0.15 * vol_norm,
                0.0,
                2.0,
            )
            df["degradation_rising_flag"] = (
                (df["elong_trend_6h"] > 0.01) | (df["elong_accel_6h"] > 0.005)
            ).astype(int)

        if temp_col in df.columns:
            df["temp_above_baseline"] = np.maximum(df[temp_col] - baseline_temp, 0.0)
            df["temp_variability_12h"] = (
                df[temp_col].rolling(window=720, min_periods=2).std().fillna(0.0)
            )
            df["temp_trend_1h"] = self._rolling_slope(df[temp_col], 60)
            df["temp_trend_6h"] = self._rolling_slope(df[temp_col], 360)
            df["temp_trend_24h"] = self._rolling_slope(df[temp_col], 1440)

            df["temp_spike_index"] = np.clip(
                (df["temp_above_baseline"] / 20.0) + (df["temp_variability_12h"] / 8.0),
                0.0,
                3.0,
            )

        if elong_col in df.columns and temp_col in df.columns:
            df["temp_elong_coupling"] = np.clip(
                (df["temp_above_baseline"] / 25.0)
                * np.clip(df["elong_trend_6h"] / 0.03, 0.0, 2.0),
                0.0,
                3.0,
            )
            df["temp_elong_interaction"] = df["temp_above_baseline"] * np.maximum(
                df["elong_delta"], 0.0
            )
            df["temp_elong_risk_flag"] = (
                (df["temp_above_baseline"] >= 15.0) & (df["elong_trend_6h"] > 0.01)
            ).astype(int)

        return df

    def create_threshold_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        sensor_cfg = self.thresholds.get("sensor_thresholds", {})

        temp_col = "temperature_boot_material/temperature_avg"
        elong_col = "ultrasonic_boot/elongation_avg"
        current_col = "current_transducer_head/current_avg"

        if temp_col in df.columns:
            tcfg = sensor_cfg.get("temperature_boot_material/temperature", {})
            df["temp_warning"] = (df[temp_col] >= float(tcfg.get("warning", 90.0))).astype(int)
            df["temp_critical"] = (df[temp_col] >= float(tcfg.get("critical", 100.0))).astype(int)

        if elong_col in df.columns:
            ecfg = sensor_cfg.get("ultrasonic_boot/elongation", {})
            df["elong_warning"] = (df[elong_col] >= float(ecfg.get("warning", 300.0))).astype(int)
            df["elong_critical"] = (df[elong_col] >= float(ecfg.get("critical", 310.0))).astype(int)

        if current_col in df.columns:
            ccfg = sensor_cfg.get("current_transducer_head/current", {})
            df["current_warning"] = (df[current_col] >= float(ccfg.get("warning", 62.0))).astype(int)
            df["current_critical"] = (df[current_col] >= float(ccfg.get("critical", 80.0))).astype(int)

        warn_cols = [c for c in ["temp_warning", "elong_warning", "current_warning"] if c in df.columns]
        crit_cols = [c for c in ["temp_critical", "elong_critical", "current_critical"] if c in df.columns]

        df["warning_count"] = df[warn_cols].sum(axis=1) if warn_cols else 0
        df["critical_count"] = df[crit_cols].sum(axis=1) if crit_cols else 0

        # 24h event intensity features (events per hour) for target shaping.
        warning_event_flag = (pd.to_numeric(df["warning_count"], errors="coerce").fillna(0.0) > 0).astype(int)
        critical_event_flag = (pd.to_numeric(df["critical_count"], errors="coerce").fillna(0.0) > 0).astype(int)

        df["warning_event_count_24h"] = (
            warning_event_flag.rolling(window=1440, min_periods=1).sum() / 60.0
        )
        df["critical_event_count_24h"] = (
            critical_event_flag.rolling(window=1440, min_periods=1).sum() / 60.0
        )

        return df

    def create_condition_state_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        warn_cols = [c for c in ["temp_warning", "elong_warning", "current_warning"] if c in df.columns]
        crit_cols = [c for c in ["temp_critical", "elong_critical", "current_critical"] if c in df.columns]

        warning_count = df[warn_cols].sum(axis=1) if warn_cols else 0
        critical_count = df[crit_cols].sum(axis=1) if crit_cols else 0
        degradation_index = pd.to_numeric(df.get("degradation_index", 0.0), errors="coerce").fillna(0.0)
        rising_flag = pd.to_numeric(df.get("degradation_rising_flag", 0), errors="coerce").fillna(0.0)
        elong_trend_6h = pd.to_numeric(df.get("elong_trend_6h", 0.0), errors="coerce").fillna(0.0)
        temp_above_baseline = pd.to_numeric(df.get("temp_above_baseline", 0.0), errors="coerce").fillna(0.0)

        df["condition_risk_score"] = np.clip(
            (0.55 * critical_count)
            + (0.20 * warning_count)
            + (0.35 * degradation_index)
            + (0.20 * rising_flag),
            0.0,
            4.0,
        )

        sustained_critical = (
            (critical_count >= 2)
            & (
                (degradation_index >= 0.80)
                | (elong_trend_6h > 0.01)
                | (temp_above_baseline >= 15.0)
            )
        )

        early_warning = (
            (warning_count >= 1)
            | (degradation_index >= 0.45)
            | (rising_flag >= 1)
            | (elong_trend_6h > 0.005)
        )

        critical_mask = sustained_critical | (df["condition_risk_score"] >= 2.20)
        warning_mask = (~critical_mask) & (early_warning | (df["condition_risk_score"] >= 0.80))

        df["condition_state_code"] = np.select([critical_mask, warning_mask], [2, 1], default=0).astype(int)
        df["is_failure"] = (df["condition_state_code"] == 2).astype(int)
        df["condition_state"] = np.select(
            [df["condition_state_code"] == 2, df["condition_state_code"] == 1],
            ["CRITICAL", "WARNING"],
            default="NORMAL",
        )

        # Explicit failure-proximity signal for the model
        df["failure_proximity"] = np.clip(
            degradation_index
            + 0.5 * pd.to_numeric(df.get("critical_count", 0), errors="coerce").fillna(0.0)
            + 0.2 * pd.to_numeric(df.get("warning_count",  0), errors="coerce").fillna(0.0),
            0.0,
            5.0,
        )

        # Fix 3: Binary failure-state trigger — 1 when critical_count≥2 AND degradation_index≥0.8
        df["failure_trigger"] = (
            (pd.to_numeric(df.get("critical_count", 0), errors="coerce").fillna(0.0) >= 2)
            & (degradation_index >= 0.8)
        ).astype(int)

        return df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Degradation-aware target generation:
        - Calendar lifecycle is the base truth
        - Exponential decay penalty based on critical events, degradation index,
          and elongation trend — so the model learns failure, not just time.
        """
        df = df.copy()

        timestamp_series = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df["@timestamp"] = timestamp_series

        baseline_rul_days = float(
            self.thresholds.get("baseline_values", {}).get("baseline_rul_days", 2190.0)
        )

        install_date = self.metadata.get("installation_date") or self.metadata.get("install_date")
        replace_date = self.metadata.get("replace_date")

        if not install_date:
            raise ValueError("installation_date/install_date missing in belts_metadata.json")
        if not replace_date:
            raise ValueError("replace_date missing in belts_metadata.json")

        install_ts = pd.to_datetime(install_date, utc=True, errors="coerce")
        replace_ts = pd.to_datetime(replace_date, utc=True, errors="coerce")

        if pd.isna(install_ts):
            raise ValueError("Invalid installation_date/install_date in belts_metadata.json")
        if pd.isna(replace_ts):
            raise ValueError("Invalid replace_date in belts_metadata.json")
        if replace_ts <= install_ts:
            raise ValueError("replace_date must be later than installation_date/install_date")

        lifecycle_days = max((replace_ts - install_ts).total_seconds() / 86400.0, 1.0)

        # Calendar RUL — stable base truth
        df["calendar_rul_days"] = (replace_ts - timestamp_series).dt.total_seconds() / 86400.0
        df["calendar_rul_days"] = np.clip(df["calendar_rul_days"], 0.0, lifecycle_days)
        
        # Hard check for NaNs
        nan_calendar = df["calendar_rul_days"].isna().sum()
        if nan_calendar > 0:
            logger.warning("Found %d NaN values in calendar_rul_days, filling with baseline", nan_calendar)
            df["calendar_rul_days"] = df["calendar_rul_days"].fillna(lifecycle_days)

        degradation_index = pd.to_numeric(df.get("degradation_index", 0.0), errors="coerce").fillna(0.0)
        warning_event_count_24h = pd.to_numeric(
            df.get("warning_event_count_24h", 0.0), errors="coerce"
        ).fillna(0.0)
        critical_event_count_24h = pd.to_numeric(
            df.get("critical_event_count_24h", 0.0), errors="coerce"
        ).fillna(0.0)

        # Target health driven directly by degradation and 24h event intensity.
        df["target_health"] = (
            100.0
            - (np.clip(degradation_index, 0.0, 2.0) * 80.0)
            - (warning_event_count_24h * 2.0)
            - (critical_event_count_24h * 5.0)
        )
        df["target_health"] = np.clip(df["target_health"], 0.0, 100.0)

        # Sharper RUL degradation target (ML-focused, not smooth calendar-only).
        rul_base_multiplier = 1.0 - np.clip(degradation_index, 0.0, 1.0)
        df["target_rul_days"] = df["calendar_rul_days"] * rul_base_multiplier
        df["target_rul_days"] = df["target_rul_days"] - (critical_event_count_24h * 50.0)
        df["target_rul_days"] = np.clip(df["target_rul_days"], 0.0, baseline_rul_days)

        # Critical label used by training for balancing/weighting.
        df["is_critical"] = (
            (df["target_health"] < 30.0)
            | (critical_event_count_24h > 0.0)
        ).astype(int)

        df["runtime_age_hours"] = (
            (timestamp_series - install_ts).dt.total_seconds() / 3600.0
        ).clip(lower=0.0)

        # Debug print
        print("\n--- Feature Engineering Debug ---")
        print(df[["calendar_rul_days", "degradation_index"]].tail())
        print("---------------------------------\n")

        return df

    def finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df = df.sort_values("@timestamp").reset_index(drop=True)
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill()

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def run(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Cleaned data not found: {file_path}")

        if self.target_mode not in {"calendar_replace_date"}:
            raise ValueError(
                f"Unsupported target_mode='{self.target_mode}'. "
                "Use 'calendar_replace_date'."
            )

        df = pd.read_csv(path)
        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["@timestamp"])

        logger.info("Running feature engineering on %d long-format rows.", len(df))

        df = self.pivot_to_wide(df)
        df = self.create_rolling_features(df)
        df = self.create_operational_features(df)
        df = self.create_degradation_features(df)
        df = self.create_threshold_indicator_features(df)
        df = self.create_condition_state_features(df)
        df = self.create_targets(df)
        df = self.finalize_features(df)

        logger.info("Feature engineering complete. Output rows: %d, columns: %d", len(df), len(df.columns))
        return df