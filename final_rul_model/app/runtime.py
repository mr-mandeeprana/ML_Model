# app/runtime.py

import logging
from datetime import datetime
from typing import Dict, Any
from app.config_loader import ConfigLoader
from app.state_manager import StateManager
from app.feature_engineering import FeatureEngineer
from app.inference_engine import InferenceEngine
from app.alert_engine import AlertEngine

logger = logging.getLogger(__name__)

class MLRuntime:
    def __init__(self):
        self.config = ConfigLoader()
        self.state_manager = StateManager()
        
        # Load configs
        thresholds = self.config.get_thresholds()
        model_cfg = self.config.get_model_config()
        
        # Initialize engines
        self.feature_engineer = FeatureEngineer(thresholds, model_cfg)
        self.inference_engine = InferenceEngine()
        self.alert_engine = AlertEngine()

    def process(self, raw_event: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single sensor event and returns predictions."""
        belt_id = raw_event.get("belt_id")
        timestamp = raw_event.get("@timestamp") or raw_event.get("timestamp") or datetime.utcnow().isoformat()
        
        if not belt_id:
            return {"status": "error", "message": "Missing belt_id"}

        try:
            # 1. Feature Engineering (Stateful)
            feature_dict = self.feature_engineer.process_event(raw_event, self.state_manager)
            feature_vector = self.feature_engineer.get_ordered_vector(feature_dict)
            
            # 2. Inference
            prediction = self.inference_engine.predict(feature_vector)
            
            # 3. Alerting Logic
            alert_status = self.alert_engine.evaluate(
                prediction["health_score"], 
                prediction["rul_days"]
            )
            
            # 4. Update State with latest metrics
            state = self.state_manager.get_state(belt_id)
            state["health_score"] = prediction["health_score"]
            state["rul_days"] = prediction["rul_days"]
            state["risk_level"] = alert_status["risk_level"]
            state["last_prediction_timestamp"] = timestamp
            
            # 5. Return combined result
            return {
                "belt_id": belt_id,
                "@timestamp": timestamp,
                "timestamp": timestamp,
                "sensor_id": raw_event.get("sensorid"),
                "health_score": prediction["health_score"],
                "rul_days": prediction["rul_days"],
                "risk_level": alert_status["risk_level"],
                "is_critical": alert_status["is_critical"],
                "is_warning": alert_status["is_warning"],
                "status": "success",
                "features_sampled": {k: feature_dict[k] for k in list(feature_dict.keys())[:5]} 
            }
            
        except Exception as e:
            logger.error(f"Runtime error processing belt {belt_id}: {e}")
            return {"status": "error", "message": str(e), "belt_id": belt_id}
