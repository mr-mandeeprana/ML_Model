"""
ML Model Training for Belt Health and RUL Prediction
Trains RandomForest models with chronological validation and strict feature checks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

logger = logging.getLogger(__name__)


class MLTrainer:
    def __init__(self, config_path: str = "config/model_config.json") -> None:
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.scaler = StandardScaler()
        self.health_model: RandomForestRegressor | None = None
        self.rul_model: RandomForestRegressor | None = None

    def _load_config(self) -> Dict:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "model_type": "random_forest",
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 3,
            "cv_n_estimators": 80,
            "retrain_on_full_data": True,
            "features": [],
            "artifacts_dir": "models/saved_models",
            "fail_on_missing_features": True,
        }

    def _build_model(self, n_estimators: int | None = None) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=int(n_estimators if n_estimators is not None else self.config.get("n_estimators", 300)),
            max_depth=self.config.get("max_depth", 20),
            min_samples_split=int(self.config.get("min_samples_split", 5)),
            min_samples_leaf=int(self.config.get("min_samples_leaf", 2)),
            max_features=self.config.get("max_features", "sqrt"),
            random_state=int(self.config.get("random_state", 42)),
            n_jobs=-1,
        )

    def _calculate_adjusted_r2(self, r2: float, n_samples: int, n_features: int) -> float:
        if n_samples <= n_features + 1:
            return r2
        adj_r2 = 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        return float(max(-1.0, min(1.0, adj_r2)))

    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")

        df = pd.read_csv(path)
        if "@timestamp" not in df.columns:
            raise ValueError("Features file must contain '@timestamp' column.")

        df["@timestamp"] = pd.to_datetime(df["@timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["@timestamp"]).sort_values("@timestamp").reset_index(drop=True)

        required_targets = ["target_health", "target_rul_days"]
        missing_targets = [c for c in required_targets if c not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        configured_features = self.config.get("features", [])
        if not configured_features:
            raise ValueError("No features defined in config/model_config.json")

        missing_features = [c for c in configured_features if c not in df.columns]
        if missing_features:
            message = f"Configured features missing in dataset: {missing_features}"
            if bool(self.config.get("fail_on_missing_features", True)):
                raise ValueError(message)
            logger.warning(message)

        selected_features = [
            c for c in configured_features
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not selected_features:
            raise ValueError("No valid numeric configured features found in dataset.")

        df = df.dropna(subset=required_targets).reset_index(drop=True)

        # Sanity Checks on Targets
        if (df["target_rul_days"] < 0).any():
            raise ValueError("Negative target_rul_days detected after target creation.")

        if (df["target_health"] < 0).any() or (df["target_health"] > 100.0).any():
            raise ValueError(f"target_health outside [0, 100] detected (min={df['target_health'].min():.2f}, max={df['target_health'].max():.2f}).")

        logger.info("Target distribution:")
        logger.info("\n%s", df[["target_health", "target_rul_days"]].describe())

        return df, selected_features

    def _calculate_sample_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Strong explicit sample weights:
        WARNING = 3x, CRITICAL = 8x, plus a degradation index boost.
        Forces the model to learn failure behavior.
        """
        if "is_critical_label" in df.columns:
            critical_flags = pd.to_numeric(df["is_critical_label"], errors="coerce").fillna(0).astype(int).values
            codes = np.where(critical_flags == 1, 2, 0)
        elif "condition_state_code" in df.columns:
            codes = pd.to_numeric(df["condition_state_code"], errors="coerce").fillna(0).astype(int).values
        else:
            return np.ones(len(df))

        weights = np.ones(len(df))

        weights[codes == 1] = 3.0    # WARNING
        weights[codes == 2] = 15.0   # CRITICAL — heavy focus on failure region

        # Extra boost proportional to degradation severity
        if "degradation_index" in df.columns:
            degrad = pd.to_numeric(df["degradation_index"], errors="coerce").fillna(0.0).values
            weights *= (1.0 + np.clip(degrad, 0.0, 2.0))

        return weights

    def _time_holdout_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        test_size = float(self.config.get("test_size", 0.2))
        test_size = min(max(test_size, 0.05), 0.5)

        n_rows = len(df)
        if n_rows < 20:
            raise ValueError("Not enough rows for reliable split. Need at least 20 rows.")

        split_type = self.config.get("validation_split_type", "chronological")
 
        if split_type != "chronological":
            raise ValueError("Only chronological validation is allowed for this time-series pipeline.")
 
        split_idx = int(n_rows * (1.0 - test_size))
        split_idx = min(max(split_idx, 1), n_rows - 1)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        logger.info("Using CHRONOLOGICAL split (test_size=%.2f)", test_size)
 
        return train_df, test_df

    def _time_series_cv_metrics(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        train_df: pd.DataFrame,
        label: str,
    ) -> Dict[str, float]:
        cv_folds = int(self.config.get("cv_folds", 0))
        if cv_folds < 2:
            return {}

        n_samples = len(X_train)
        if n_samples < 30:
            logger.warning("Skipping %s CV: not enough rows (%d).", label, n_samples)
            return {}

        effective_splits = min(cv_folds, max(2, n_samples - 1))
        tscv = TimeSeriesSplit(n_splits=effective_splits)

        fold_mae: List[float] = []
        fold_rmse: List[float] = []
        fold_r2: List[float] = []

        cv_n_estimators = int(self.config.get("cv_n_estimators", 80))
        cv_n_estimators = max(10, cv_n_estimators)

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
 
            # Use consistent sample weighting policy for each CV fold
            fold_df = train_df.iloc[tr_idx].copy()
            fold_weights = self._calculate_sample_weights(fold_df)
 
            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)
 
            fold_model = self._build_model(n_estimators=cv_n_estimators)
            fold_model.fit(X_tr_scaled, y_tr, sample_weight=fold_weights)
            y_pred = fold_model.predict(X_val_scaled)
 
            fold_mae.append(float(mean_absolute_error(y_val, y_pred)))
            fold_rmse.append(float(np.sqrt(mean_squared_error(y_val, y_pred))))
            fold_r2.append(float(r2_score(y_val, y_pred)))
 
            logger.info("%s CV fold %d/%d complete.", label, fold_idx, effective_splits)

        return {
            f"{label}_cv_folds": int(effective_splits),
            f"{label}_cv_mae_mean": float(np.mean(fold_mae)),
            f"{label}_cv_mae_std": float(np.std(fold_mae)),
            f"{label}_cv_rmse_mean": float(np.mean(fold_rmse)),
            f"{label}_cv_rmse_std": float(np.std(fold_rmse)),
            f"{label}_cv_r2_mean": float(np.mean(fold_r2)),
            f"{label}_cv_r2_std": float(np.std(fold_r2)),
        }

    def _evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_features: int,
        prefix: str,
    ) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        adj_r2 = self._calculate_adjusted_r2(r2, len(y_true), n_features)

        return {
            f"{prefix}_mae": mae,
            f"{prefix}_rmse": rmse,
            f"{prefix}_r2": r2,
            f"{prefix}_adj_r2": adj_r2,
        }

    def _evaluate_by_condition_state(
        self,
        test_df: pd.DataFrame,
        h_test_preds: np.ndarray,
        r_test_preds: np.ndarray,
    ) -> Dict[str, float]:
        if "condition_state_code" not in test_df.columns:
            return {}

        segment_map = {
            0: "normal",
            1: "warning",
            2: "critical",
        }

        results: Dict[str, float] = {}

        for state_code, segment_name in segment_map.items():
            mask = pd.to_numeric(test_df["condition_state_code"], errors="coerce").fillna(-1).astype(int) == state_code
            n_segment = int(mask.sum())
            results[f"segment_{segment_name}_n"] = n_segment

            if n_segment < 2:
                continue

            y_health = test_df.loc[mask, "target_health"].values
            y_rul = test_df.loc[mask, "target_rul_days"].values
            h_pred = h_test_preds[mask.values]
            r_pred = r_test_preds[mask.values]

            results[f"segment_{segment_name}_health_mae"] = float(mean_absolute_error(y_health, h_pred))
            results[f"segment_{segment_name}_rul_mae"] = float(mean_absolute_error(y_rul, r_pred))

            try:
                results[f"segment_{segment_name}_health_r2"] = float(r2_score(y_health, h_pred))
            except Exception:
                pass

            try:
                results[f"segment_{segment_name}_rul_r2"] = float(r2_score(y_rul, r_pred))
            except Exception:
                pass

        return results

    def _save_models(self, features: List[str], metrics: Dict[str, float]) -> None:
        artifacts_dir = Path(self.config.get("artifacts_dir", "models/saved_models"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if self.health_model is None or self.rul_model is None:
            raise ValueError("Models are not trained yet.")

        health_feature_importance = {}
        rul_feature_importance = {}

        if hasattr(self.health_model, "feature_importances_"):
            health_feature_importance = {
                feature: float(score)
                for feature, score in zip(features, self.health_model.feature_importances_)
            }

        if hasattr(self.rul_model, "feature_importances_"):
            rul_feature_importance = {
                feature: float(score)
                for feature, score in zip(features, self.rul_model.feature_importances_)
            }

        health_artifact = {
            "model": self.health_model,
            "scaler": self.scaler,
            "features": features,
            "config": self.config,
            "metrics": metrics,
            "feature_importance": health_feature_importance,
            "target_name": "target_health",
        }

        rul_artifact = {
            "model": self.rul_model,
            "scaler": self.scaler,
            "features": features,
            "config": self.config,
            "metrics": metrics,
            "feature_importance": rul_feature_importance,
            "target_name": "target_rul_days",
            "target_transform": "log1p",
        }

        joblib.dump(health_artifact, artifacts_dir / "ml_health_model.pkl", compress=3)
        joblib.dump(rul_artifact, artifacts_dir / "ml_rul_model.pkl", compress=3)

        logger.info("Saved health model to %s", artifacts_dir / "ml_health_model.pkl")
        logger.info("Saved RUL model to %s", artifacts_dir / "ml_rul_model.pkl")

    def train_and_validate(self, data_path: str) -> Dict[str, float]:
        df, features = self.load_data(data_path)

        # Unified critical label for robust training balance.
        if "is_critical" in df.columns:
            critical_mask = pd.to_numeric(df["is_critical"], errors="coerce").fillna(0).astype(int) == 1
        else:
            critical_mask = pd.Series(False, index=df.index)

        if "condition_state_code" in df.columns:
            state_critical = pd.to_numeric(df["condition_state_code"], errors="coerce").fillna(0).astype(int) == 2
            critical_mask = critical_mask | state_critical

        if "critical_event_count_24h" in df.columns:
            event_critical = pd.to_numeric(df["critical_event_count_24h"], errors="coerce").fillna(0.0) > 0.0
            critical_mask = critical_mask | event_critical

        # Keep explicit label for downstream weighting and diagnostics.
        df["is_critical_label"] = critical_mask.astype(int)

        total_critical = int(df["is_critical_label"].sum())
        total_warning = int(df["condition_state_code"].eq(1).sum()) if "condition_state_code" in df.columns else 0
        low_failure_mode = total_critical < 50

        if low_failure_mode:
            logger.warning(
                "Low critical sample count detected (%d). Training will continue with balancing.",
                total_critical,
            )
            print(f"\n[WARNING] Low critical samples ({total_critical}), but training continues with balancing...")
        else:
            logger.info(
                "Failure-rich dataset detected: %d critical, %d warning rows.",
                total_critical,
                total_warning,
            )

        train_df, test_df = self._time_holdout_split(df)

        # Fix 1: Hard validation
        nan_ratio = train_df[features].isna().sum().sum() / (len(train_df) * len(features))
        if nan_ratio > 0.01:
            raise ValueError(f"Too many NaNs in training features: {nan_ratio:.4f}")

        train_df[features] = train_df[features].ffill().bfill()
        test_df[features] = test_df[features].ffill().bfill()

        # Balance training data: upsample critical class to ~50% of normal class.
        critical_df = train_df[train_df["is_critical_label"] == 1].copy()
        normal_df = train_df[train_df["is_critical_label"] == 0].copy()

        if len(critical_df) > 0 and len(normal_df) > 0:
            target_critical = max(int(len(normal_df) // 2), len(critical_df))
            critical_upsampled = resample(
                critical_df,
                replace=True,
                n_samples=target_critical,
                random_state=int(self.config.get("random_state", 42)),
            )
            train_df = pd.concat([normal_df, critical_upsampled], ignore_index=True)
            train_df = train_df.sort_values("@timestamp").reset_index(drop=True)
            logger.info(
                "Balanced training dataset prepared: normal=%d, critical=%d (upsampled to %d). Total=%d",
                len(normal_df),
                len(critical_df),
                target_critical,
                len(train_df),
            )
        else:
            logger.warning(
                "Critical balancing for training skipped (normal=%d, critical=%d).",
                len(normal_df),
                len(critical_df),
            )

        # Always apply weighting; this includes degradation-aware boosts.
        train_weights = self._calculate_sample_weights(train_df)

        if train_df[features].isna().any().any():
            raise ValueError("NaN found in training features. Fix feature engineering before training.")

        if test_df[features].isna().any().any():
            raise ValueError("NaN found in test features. Fix feature engineering before validation.")

        X_train = train_df[features].values
        X_test = test_df[features].values

        y_train_health = train_df["target_health"].values
        y_test_health = test_df["target_health"].values

        y_train_rul = np.log1p(train_df["target_rul_days"].values)
        y_test_rul = np.log1p(test_df["target_rul_days"].values)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        logger.info("Training Health model (with sample weighting)...")
        self.health_model = self._build_model()
        self.health_model.fit(X_train_scaled, y_train_health, sample_weight=train_weights)

        logger.info("Training RUL model on log1p-scaled targets (with sample weighting)...")
        self.rul_model = self._build_model()
        self.rul_model.fit(X_train_scaled, y_train_rul, sample_weight=train_weights)

        # Cross-Validation with aligned weights (CV uses log-space targets already)
        cv_metrics = {}
        if int(self.config.get("cv_folds", 0)) >= 2:
            logger.info("Running time-series CV on training window...")
            cv_metrics.update(self._time_series_cv_metrics(X_train, y_train_health, train_df, "health"))
            cv_metrics.update(self._time_series_cv_metrics(X_train, y_train_rul, train_df, "rul"))

        # --- Feature Selection (importance-based) ---
        if self.health_model is not None and hasattr(self.health_model, "feature_importances_"):
            importances = self.health_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                "feature": features,
                "importance": importances
            }).sort_values("importance", ascending=False)

            top_features = feature_importance_df.head(40)["feature"].tolist()
            logger.info("Selected top 40 features based on initial training importance.")
            
            # Update the features list and re-prepare arrays
            features = top_features
            self.config["features"] = features
            
            X_train = train_df[features].values
            X_test = test_df[features].values
            
            # Re-fit scaler and models on reduced feature set
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled  = self.scaler.transform(X_test)

            logger.info("Retraining models on reduced 40-feature set...")
            self.health_model = self._build_model()
            self.health_model.fit(X_train_scaled, y_train_health, sample_weight=train_weights)

            self.rul_model = self._build_model()
            self.rul_model.fit(X_train_scaled, y_train_rul, sample_weight=train_weights)

        # --- Re-calculate Metrics on reduced set ---
        h_train_preds = self.health_model.predict(X_train_scaled)
        h_test_preds  = self.health_model.predict(X_test_scaled)
        r_train_preds_log = self.rul_model.predict(X_train_scaled)
        r_test_preds_log = self.rul_model.predict(X_test_scaled)

        r_train_preds = np.expm1(r_train_preds_log)
        r_test_preds = np.expm1(r_test_preds_log)
        y_rul_train_real = np.expm1(y_train_rul)
        y_rul_test_real  = np.expm1(y_test_rul)

        n_features = len(features)
        metrics: Dict[str, float] = {
            "n_total":          int(len(df)),
            "n_train":          int(len(train_df)),
            "n_test":           int(len(test_df)),
            "n_features":       int(n_features),
            "split_time_utc":   str(test_df["@timestamp"].iloc[0]),
            "low_failure_mode": int(low_failure_mode),
            "total_critical":   total_critical,
            "total_warning":    total_warning,
        }

        metrics.update(self._evaluate_predictions(y_train_health, h_train_preds, n_features, "health_train"))
        metrics.update(self._evaluate_predictions(y_test_health,  h_test_preds,  n_features, "health_test"))
        metrics.update(self._evaluate_predictions(y_rul_train_real, r_train_preds, n_features, "rul_train"))
        metrics.update(self._evaluate_predictions(y_rul_test_real,  r_test_preds,  n_features, "rul_test"))
        metrics.update(self._evaluate_by_condition_state(test_df, h_test_preds, r_test_preds))
        metrics.update(cv_metrics)

        # Final full retraining if requested
        if bool(self.config.get("retrain_on_full_data", True)):
            logger.info("Final retraining on balanced full dataset (reduced feature set)...")
            df_all = df.copy()
            all_weights = self._calculate_sample_weights(df_all)
            X_all = df_all[features].values
            y_health_all = df_all["target_health"].values
            y_rul_all    = np.log1p(df_all["target_rul_days"].values)

            self.scaler.fit(X_all)
            X_all_scaled = self.scaler.transform(X_all)

            self.health_model = self._build_model()
            self.health_model.fit(X_all_scaled, y_health_all, sample_weight=all_weights)
            self.rul_model = self._build_model()
            self.rul_model.fit(X_all_scaled, y_rul_all, sample_weight=all_weights)

        self._save_models(features, metrics)
        return metrics