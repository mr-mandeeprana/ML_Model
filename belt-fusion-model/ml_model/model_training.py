"""
ML Model Training for Belt Health and RUL Prediction
Trains RandomForest models to predict Health Score [0, 100] and RUL [0, 5000].
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import json
import logging
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

class MLTrainer:
    def __init__(self, config_path: str = "config/model_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.scaler = StandardScaler()
        self.health_model = None
        self.rul_model = None
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "model_type": "random_forest",
            "n_estimators": 50,
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "random_state": 42,
            "test_size": 0.2,
            "cv_folds": 3,
            "cv_n_estimators": 80,
            "retrain_on_full_data": True
        }

    def _build_model(self, n_estimators: int = None) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=int(n_estimators if n_estimators is not None else self.config.get('n_estimators', 30)),
            max_depth=self.config.get('max_depth', 5),
            min_samples_split=int(self.config.get('min_samples_split', 2)),
            min_samples_leaf=int(self.config.get('min_samples_leaf', 1)),
            max_features=self.config.get('max_features', 'sqrt'),
            random_state=int(self.config.get('random_state', 42)),
            n_jobs=-1
        )
    
    def _calculate_adjusted_r2(self, r2: float, n_samples: int, n_features: int) -> float:
        """
        Calculate adjusted R² which penalizes for adding more features.
        Adj-R² = 1 - ((1 - R²) * (n - 1) / (n - p - 1))
        where n = sample count, p = feature count
        """
        if n_samples <= n_features + 1:
            return r2  # Not enough samples to adjust
        adj_r2 = 1.0 - ((1.0 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        return max(-1.0, min(1.0, adj_r2))  # Clamp to [-1, 1]
    
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        df = df.sort_values('@timestamp')
        
        exclude = ['@timestamp', 'target_health', 'target_rul_days', 'ml_penalty', 'asset_id', 'failure_type', 'sensorid', 'operating_hours', 'calendar_hours']
        configured_features = self.config.get('features', [])
        if isinstance(configured_features, list) and configured_features:
            features = [
                c for c in configured_features
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            missing = [c for c in configured_features if c not in df.columns]
            if missing:
                logger.warning("Configured features missing in dataset and skipped: %s", missing)
        else:
            features = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

        if not features:
            raise ValueError("No valid numeric features selected. Check config/model_config.json features.")
        
        df = df.dropna(subset=['target_health', 'target_rul_days'])
        return df, features

    def _time_holdout_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Chronological split: train on past, test on future to avoid leakage."""
        test_size = float(self.config.get('test_size', 0.2))
        test_size = min(max(test_size, 0.05), 0.5)

        n_rows = len(df)
        if n_rows < 10:
            raise ValueError("Not enough rows for reliable train/test split (need at least 10).")

        split_idx = int(n_rows * (1.0 - test_size))
        split_idx = min(max(split_idx, 1), n_rows - 1)

        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        return train_df, test_df

    def _time_series_cv_metrics(self, X_train: np.ndarray, y_train: np.ndarray, label: str) -> Dict[str, float]:
        """Time-ordered CV on train segment only (no future leakage)."""
        cv_folds = int(self.config.get('cv_folds', 0))
        if cv_folds < 2:
            return {}

        n_samples = len(X_train)
        if n_samples < 20:
            logger.warning("Skipping %s CV: not enough training rows (%d).", label, n_samples)
            return {}

        effective_splits = min(cv_folds, max(2, n_samples - 1))
        tscv = TimeSeriesSplit(n_splits=effective_splits)

        fold_mae = []
        fold_rmse = []
        fold_r2 = []

        cv_n_estimators = int(self.config.get('cv_n_estimators', min(100, int(self.config.get('n_estimators', 30)))))
        cv_n_estimators = max(10, min(cv_n_estimators, int(self.config.get('n_estimators', 30))))

        for fold_idx, (tr_idx, val_idx) in enumerate(tscv.split(X_train), start=1):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            fold_scaler = StandardScaler()
            X_tr_scaled = fold_scaler.fit_transform(X_tr)
            X_val_scaled = fold_scaler.transform(X_val)

            fold_model = self._build_model(n_estimators=cv_n_estimators)
            fold_model.fit(X_tr_scaled, y_tr)
            y_pred = fold_model.predict(X_val_scaled)

            fold_mae.append(mean_absolute_error(y_val, y_pred))
            fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_pred)))
            fold_r2.append(r2_score(y_val, y_pred))
            logger.info("%s CV fold %d/%d complete.", label, fold_idx, effective_splits)

        return {
            f'{label}_cv_folds': int(effective_splits),
            f'{label}_cv_mae_mean': float(np.mean(fold_mae)),
            f'{label}_cv_mae_std': float(np.std(fold_mae)),
            f'{label}_cv_rmse_mean': float(np.mean(fold_rmse)),
            f'{label}_cv_rmse_std': float(np.std(fold_rmse)),
            f'{label}_cv_r2_mean': float(np.mean(fold_r2)),
            f'{label}_cv_r2_std': float(np.std(fold_r2))
        }
    
    def train_and_validate(self, data_path: str) -> Dict[str, float]:
        df, features = self.load_data(data_path)
        train_df, test_df = self._time_holdout_split(df)

        X_train = train_df[features].fillna(0).values
        X_test = test_df[features].fillna(0).values
        y_health_train = train_df['target_health'].values
        y_health_test = test_df['target_health'].values
        y_rul_train = train_df['target_rul_days'].values
        y_rul_test = test_df['target_rul_days'].values

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Training Health Model...")
        self.health_model = self._build_model()
        self.health_model.fit(X_train_scaled, y_health_train)
        
        logger.info("Training RUL Model...")
        self.rul_model = self._build_model()
        self.rul_model.fit(X_train_scaled, y_rul_train)

        cv_metrics = {}
        if int(self.config.get('cv_folds', 0)) >= 2:
            logger.info("Running time-series CV on training window...")
            cv_metrics.update(self._time_series_cv_metrics(X_train, y_health_train, 'health'))
            cv_metrics.update(self._time_series_cv_metrics(X_train, y_rul_train, 'rul'))

        # Evaluate on both train and held-out future test window
        h_train_preds = self.health_model.predict(X_train_scaled)
        h_test_preds = self.health_model.predict(X_test_scaled)
        h_train_preds = self.health_model.predict(X_train_scaled)
        h_test_preds = self.health_model.predict(X_test_scaled)
        r_train_preds = self.rul_model.predict(X_train_scaled)
        r_test_preds = self.rul_model.predict(X_test_scaled)

        # Calculate R² values
        h_train_r2 = float(r2_score(y_health_train, h_train_preds))
        h_test_r2 = float(r2_score(y_health_test, h_test_preds))
        r_train_r2 = float(r2_score(y_rul_train, r_train_preds))
        r_test_r2 = float(r2_score(y_rul_test, r_test_preds))
        
        # Calculate adjusted R² values
        n_features = len(features)
        h_train_adj_r2 = self._calculate_adjusted_r2(h_train_r2, len(y_health_train), n_features)
        h_test_adj_r2 = self._calculate_adjusted_r2(h_test_r2, len(y_health_test), n_features)
        r_train_adj_r2 = self._calculate_adjusted_r2(r_train_r2, len(y_rul_train), n_features)
        r_test_adj_r2 = self._calculate_adjusted_r2(r_test_r2, len(y_rul_test), n_features)

        metrics = {
            'n_train': int(len(train_df)),
            'n_test': int(len(test_df)),
            'n_features': int(n_features),
            'split_time_utc': str(test_df['@timestamp'].iloc[0]),
            'health_train_mae': float(mean_absolute_error(y_health_train, h_train_preds)),
            'health_train_rmse': float(np.sqrt(mean_squared_error(y_health_train, h_train_preds))),
            'health_train_r2': h_train_r2,
            'health_train_adj_r2': h_train_adj_r2,
            'health_test_mae': float(mean_absolute_error(y_health_test, h_test_preds)),
            'health_test_rmse': float(np.sqrt(mean_squared_error(y_health_test, h_test_preds))),
            'health_test_r2': h_test_r2,
            'health_test_adj_r2': h_test_adj_r2,
            'rul_train_mae': float(mean_absolute_error(y_rul_train, r_train_preds)),
            'rul_train_rmse': float(np.sqrt(mean_squared_error(y_rul_train, r_train_preds))),
            'rul_train_r2': r_train_r2,
            'rul_train_adj_r2': r_train_adj_r2,
            'rul_test_mae': float(mean_absolute_error(y_rul_test, r_test_preds)),
            'rul_test_rmse': float(np.sqrt(mean_squared_error(y_rul_test, r_test_preds))),
            'rul_test_r2': r_test_r2,
            'rul_test_adj_r2': r_test_adj_r2
        }
        metrics.update(cv_metrics)

        if bool(self.config.get('retrain_on_full_data', True)):
            logger.info("Retraining final models on full dataset...")
            X_full = df[features].fillna(0).values
            y_health_full = df['target_health'].values
            y_rul_full = df['target_rul_days'].values

            self.scaler.fit(X_full)
            X_full_scaled = self.scaler.transform(X_full)

            self.health_model = self._build_model()
            self.health_model.fit(X_full_scaled, y_health_full)

            self.rul_model = self._build_model()
            self.rul_model.fit(X_full_scaled, y_rul_full)
            metrics['retrained_on_full_data'] = True
        else:
            metrics['retrained_on_full_data'] = False
        
        self._save_models(features)
        return metrics
    
    def _save_models(self, features: List[str]):
        models_dir = Path("models/saved_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.health_model,
            'scaler': self.scaler,
            'features': features,
            'config': self.config
        }, models_dir / "ml_health_model.pkl")
        
        joblib.dump({
            'model': self.rul_model,
            'scaler': self.scaler,
            'features': features,
            'config': self.config
        }, models_dir / "ml_rul_model.pkl")
        logger.info(f"Models saved to {models_dir}")
