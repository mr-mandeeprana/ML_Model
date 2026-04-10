"""
ML Predictor
Loads trained ML models to predict Health Score and RUL (days) directly.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class MLPredictor:
    def __init__(self, models_dir: str = "models/saved_models", thresholds_config_path: str = "config/thresholds.json"):
        self.models_dir = Path(models_dir)
        self.thresholds_config_path = Path(thresholds_config_path)
        self.baseline_rul_days = self._load_baseline_rul_days()
        self.health_artifact = self._load_model("ml_health_model.pkl")
        self.rul_artifact = self._load_model("ml_rul_model.pkl")

    def _load_baseline_rul_days(self) -> float:
        if self.thresholds_config_path.exists():
            try:
                with open(self.thresholds_config_path, 'r') as f:
                    cfg = json.load(f)
                return float(cfg.get('baseline_values', {}).get('baseline_rul_days', 2190.0))
            except Exception as exc:
                logger.warning("Failed to load baseline RUL from %s: %s", self.thresholds_config_path, exc)
        return 2190.0
        
    def _load_model(self, filename: str):
        path = self.models_dir / filename
        if not path.exists():
            return None
        return joblib.load(path)
        
    def predict(self, df_features: pd.DataFrame) -> pd.DataFrame:
        df = df_features.copy()
        
        # Defaults
        df['ml_health'] = 90.0
        df['ml_rul_days'] = 2190.0
        
        if self.health_artifact is None or self.rul_artifact is None:
            logger.warning("ML models not fully loaded! Returning defaults.")
            return df
            
        scaler = self.health_artifact['scaler'] # same scaler for both
        features = self.health_artifact.get('features', [])
        
        X_df = pd.DataFrame(index=df.index)
        for col in features:
            X_df[col] = df.get(col, 0.0).fillna(0.0)
            
        X_scaled = scaler.transform(X_df.values)
        
        h_model = self.health_artifact['model']
        r_model = self.rul_artifact['model']
        
        health_preds = h_model.predict(X_scaled)
        rul_preds = r_model.predict(X_scaled)
        
        df['ml_health'] = np.clip(health_preds, 0.0, 100.0)
        # Cap ML RUL to configured baseline life to keep outputs physically bounded.
        df['ml_rul_days'] = np.clip(rul_preds, 0.0, self.baseline_rul_days)
        
        return df
