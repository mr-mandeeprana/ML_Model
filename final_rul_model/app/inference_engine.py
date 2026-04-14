# app/inference_engine.py

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = Path(model_dir)
        self.health_model = self._load_model("ml_health_model.pkl")
        self.rul_model = self._load_model("ml_rul_model.pkl")
        
        # Extract features and scaler from artifacts
        if self.health_model:
            self.scaler = self.health_model.get("scaler")
            self.model_health = self.health_model.get("model")
        
        if self.rul_model:
            self.model_rul = self.rul_model.get("model")

    def _load_model(self, filename: str) -> Optional[Dict[str, Any]]:
        path = self.model_dir / filename
        if not path.exists():
            logger.error(f"Model file not found: {path}")
            return None
        
        import time
        start_time = time.time()
        try:
            model = joblib.load(path)
            duration = time.time() - start_time
            size_mb = path.stat().st_size / (1024 * 1024)
            logger.info(f"Loaded model {filename} ({size_mb:.1f} MB) in {duration:.2f}s")
            return model
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def predict(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Runs both Health and RUL models."""
        if not self.health_model or not self.rul_model:
            return {"health_score": 90.0, "rul_days": 2190.0, "status": "fallback"}

        try:
            # 1. Scale
            X_scaled = self.scaler.transform(feature_vector)
            
            # 2. Predict Health
            health_score = float(self.model_health.predict(X_scaled)[0])
            health_score = np.clip(health_score, 0.0, 100.0)
            
            # 3. Predict RUL
            rul_days = float(self.model_rul.predict(X_scaled)[0])
            rul_days = np.clip(rul_days, 0.0, 2190.0)
            
            return {
                "health_score": round(health_score, 2),
                "rul_days": round(rul_days, 1),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {"health_score": 75.0, "rul_days": 1800.0, "status": "error", "error": str(e)}
