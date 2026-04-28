"""
Runtime Engine
Orchestrates streaming processing:
state → features → inference → alerts
"""

import logging
import os
from typing import Dict, Any

from app.state_manager import StateManager
from app.feature_engineering import FeatureBuilder
from app.inference_engine import InferenceEngine
from app.alert_engine import AlertEngine
from app.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class RuntimeEngine:
    def __init__(self):
        logger.info("Initializing Runtime Engine...")

        # Load config
        self.config = ConfigLoader()

        # Core components
        self.state_manager = StateManager()
        self.feature_builder = FeatureBuilder(self.config)
        self.inference_engine = InferenceEngine(
            model_dir=os.getenv("MODEL_DIR", "models/saved_models")
        )
        self.alert_engine = AlertEngine(self.config)

        logger.info("Runtime Engine ready.")

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one streaming event
        """

        try:
            # 1. Update state
            state = self.state_manager.update(event)

            # If not enough data yet (cold start), skip
            if not state.get("ready", False):
                return None

            # 2. Build features
            features = self.feature_builder.build(state)

            if features is None:
                return None

            # 3. Run inference
            prediction = self.inference_engine.predict(features)

            # 4. Generate alerts
            result = self.alert_engine.apply(prediction)

            # 5. Attach metadata and new ML signals
            result["@timestamp"] = event.get("@timestamp")
            result["sensorid"] = event.get("sensorid")
            result["belt_id"] = event.get("belt_id", "belt-1")
            
            # Propagation of rich ML metrics
            result["confidence_level"] = prediction.get("confidence_level", "LOW")
            result["prediction_status"] = prediction.get("prediction_status", "unknown")
            result["failure_risk_score"] = prediction.get("failure_risk_score", 0.0)
            result["early_failure_flag"] = prediction.get("early_failure_flag", 0)

            return result

        except Exception as e:
            logger.error(f"Runtime processing failed: {e}")
            return None