"""
Data Preprocessing Module
Loads, validates, cleans, and aligns raw sensor data for ML training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Prepares raw long-format sensor data for downstream feature engineering.

    Expected input columns:
    - sensorid
    - @timestamp
    - avg_value

    Optional columns:
    - max_value
    - min_value
    - std_deviation
    """

    CORE_SENSORS = [
        "temperature_boot_material/temperature",
        "ultrasonic_boot/elongation",
        "current_transducer_head/current",
    ]

    def __init__(self, config_path: str = "config/thresholds.json", resample_freq: str = "1min") -> None:
        self.config_path = Path(config_path)
        self.resample_freq = resample_freq
        self.thresholds = self._load_config()

    def _load_config(self) -> Dict:
        if not self.config_path.exists():
            logger.warning("Config not found at %s. Using empty defaults.", self.config_path)
            return {}

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Error loading config from %s: %s", self.config_path, exc)
            raise

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required_cols = ["sensorid", "@timestamp", "avg_value"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def load_sensor_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_csv(path, on_bad_lines="skip")
        except Exception as exc:
            raise ValueError(f"Failed to read CSV: {exc}") from exc

        self._validate_columns(df)

        df["sensorid"] = df["sensorid"].astype(str).str.strip()
        df = df[df["sensorid"].isin(self.CORE_SENSORS)].copy()

        if df.empty:
            raise ValueError(
                "No valid core sensor data found. Required sensors: "
                + ", ".join(self.CORE_SENSORS)
            )

        df["@timestamp"] = pd.to_datetime(df["@timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["@timestamp"])

        numeric_cols = ["avg_value", "min_value", "max_value", "std_deviation"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["avg_value"])

        return df.sort_values(["@timestamp", "sensorid"]).reset_index(drop=True)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        before = len(df)
        df = df.drop_duplicates(subset=["sensorid", "@timestamp"], keep="last")

        sensor_thresholds = self.thresholds.get("sensor_thresholds", {})
        cleaned_parts: List[pd.DataFrame] = []

        for sensor in self.CORE_SENSORS:
            sensor_df = df[df["sensorid"] == sensor].copy()
            if sensor_df.empty:
                continue

            limits = sensor_thresholds.get(sensor, {})
            min_val = limits.get("min", -np.inf)
            max_val = limits.get("max", np.inf)

            sensor_df = sensor_df[
                sensor_df["avg_value"].between(min_val, max_val, inclusive="both")
            ].copy()

            cleaned_parts.append(sensor_df)

        if not cleaned_parts:
            raise ValueError("All rows were removed during cleaning.")

        cleaned_df = pd.concat(cleaned_parts, ignore_index=True)
        cleaned_df = cleaned_df.sort_values(["@timestamp", "sensorid"]).reset_index(drop=True)

        logger.info(
            "Cleaning complete: kept %d/%d rows after duplicates and range filtering.",
            len(cleaned_df),
            before,
        )
        return cleaned_df

    def resample_to_minute_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample each core sensor to a consistent 1-minute grid.
        This keeps long-format output, which matches the next pipeline stage.
        """
        parts: List[pd.DataFrame] = []

        for sensor in self.CORE_SENSORS:
            sensor_df = df[df["sensorid"] == sensor].copy()
            if sensor_df.empty:
                logger.warning("Missing sensor in cleaned data: %s", sensor)
                continue

            sensor_df = sensor_df.sort_values("@timestamp").set_index("@timestamp")

            resampled = sensor_df[["avg_value"]].resample(self.resample_freq).mean()

            if "min_value" in sensor_df.columns:
                resampled["min_value"] = sensor_df["min_value"].resample(self.resample_freq).mean()
            else:
                resampled["min_value"] = np.nan

            if "max_value" in sensor_df.columns:
                resampled["max_value"] = sensor_df["max_value"].resample(self.resample_freq).mean()
            else:
                resampled["max_value"] = np.nan

            if "std_deviation" in sensor_df.columns:
                resampled["std_deviation"] = sensor_df["std_deviation"].resample(self.resample_freq).mean()
            else:
                resampled["std_deviation"] = np.nan

            resampled["avg_value"] = resampled["avg_value"].interpolate(limit_direction="both")
            resampled["min_value"] = resampled["min_value"].interpolate(limit_direction="both")
            resampled["max_value"] = resampled["max_value"].interpolate(limit_direction="both")
            resampled["std_deviation"] = resampled["std_deviation"].fillna(0.0)

            resampled["sensorid"] = sensor
            resampled = resampled.reset_index()

            parts.append(resampled)

        if not parts:
            raise ValueError("No sensor data available after resampling.")

        out = pd.concat(parts, ignore_index=True)
        out = out.sort_values(["@timestamp", "sensorid"]).reset_index(drop=True)
        return out

    def validate_sensor_coverage(self, df: pd.DataFrame) -> None:
        present = set(df["sensorid"].unique())
        missing = [sensor for sensor in self.CORE_SENSORS if sensor not in present]
        if missing:
            raise ValueError(f"Missing required sensors after preprocessing: {missing}")

    def preprocess(self, file_path: str) -> pd.DataFrame:
        logger.info("Starting preprocessing for %s", file_path)

        df = self.load_sensor_data(file_path)
        self.validate_sensor_coverage(df)

        df = self.clean_data(df)
        self.validate_sensor_coverage(df)

        df = self.resample_to_minute_grid(df)
        self.validate_sensor_coverage(df)

        logger.info("Preprocessing complete. Final rows: %d", len(df))
        return df