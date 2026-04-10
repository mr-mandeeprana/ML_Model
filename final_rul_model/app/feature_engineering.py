# app/feature_engineering.py

import logging
import numpy as np
from typing import Dict, Any, List
from app.state_manager import StateManager

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, thresholds: Dict[str, Any], model_config: Dict[str, Any]):
        self.thresholds = thresholds
        self.required_features = model_config.get("features", [])
        self.sensor_warn_crit = thresholds.get("sensor_warning_critical", {})
        
        # Mapping base sensor IDs to their config names
        self.sensor_map = {
            "current_transducer": "current_transducer_head/current",
            "temperature_boot_material": "temperature_boot_material/temperature",
            "ultrasonic_boot": "ultrasonic_boot/elongation"
        }

    def derive_flags(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Compute warning and critical flags based on thresholds."""
        flags = {}
        sensor_id = event.get("sensorid")
        val = event.get("avg_value", event.get("value"))
        
        if sensor_id in self.sensor_map:
            cfg_name = self.sensor_map[sensor_id]
            t_cfg = self.sensor_warn_crit.get(cfg_name, {})
            
            warn = t_cfg.get("warning")
            crit = t_cfg.get("critical")
            
            if warn is not None:
                flags[f"{sensor_id}_warning"] = int(val >= warn)
            if crit is not None:
                flags[f"{sensor_id}_critical"] = int(val >= crit)
        
        return flags

    def process_event(self, event: Dict[str, Any], state_manager: StateManager) -> Dict[str, float]:
        """
        Main entry for streaming feature engineering.
        Updates state and returns the full feature dictionary.
        """
        belt_id = event.get("belt_id")
        sensor_id = event.get("sensorid")
        val = float(event.get("avg_value", event.get("value", 0.0)))
        
        # 1. Update rolling buffer in state
        if sensor_id in ["current_transducer", "temperature_boot_material", "ultrasonic_boot"]:
            state_manager.update_buffer(belt_id, sensor_id, val)
        
        # 2. Increment operating hours if not idle
        idle_thresh = self.thresholds.get("idle_detection", {}).get("current_threshold", 4.5)
        is_idle = 0
        if sensor_id == "current_transducer":
            is_idle = int(val < idle_thresh)
            if not is_idle:
                state_manager.increment_operating_hours(belt_id)
        
        # 3. Build features
        state = state_manager.get_state(belt_id)
        features = {}
        
        # Current Value placeholders
        features["current_transducer_head/current_avg"] = val if sensor_id == "current_transducer" else state.get("last_current", 0.0)
        features["temperature_boot_material/temperature_avg"] = val if sensor_id == "temperature_boot_material" else state.get("last_temp", 0.0)
        features["ultrasonic_boot/elongation_avg"] = val if sensor_id == "ultrasonic_boot" else state.get("last_elong", 0.0)
        
        # Update last seen values in state
        if sensor_id == "current_transducer": state["last_current"] = val
        if sensor_id == "temperature_boot_material": state["last_temp"] = val
        if sensor_id == "ultrasonic_boot": state["last_elong"] = val

        # Rolling Stats (1h, 12h)
        for sid, prefix in [("current_transducer", "current_transducer_head/current"), 
                            ("temperature_boot_material", "temperature_boot_material/temperature"), 
                            ("ultrasonic_boot", "ultrasonic_boot/elongation")]:
            for window in [60, 720]:
                w_name = f"{window // 60}h"
                stats = state_manager.get_buffer_stats(belt_id, sid, window)
                features[f"{prefix}_{w_name}_mean"] = stats["mean"]
                features[f"{prefix}_{w_name}_std"] = stats["std"]
                features[f"{prefix}_{w_name}_min"] = stats["min"]
                features[f"{prefix}_{w_name}_max"] = stats["max"]

        # Operational/Degradation
        features["is_idle"] = is_idle
        features["operating_hours"] = state.get("operating_hours", 0.0)
        
        # Simplified versions of complex features for streaming
        features["high_load"] = int(features["current_transducer_head/current_avg"] >= 62.0)
        features["current_percentile"] = 50.0 # Placeholder or compute from buffer
        
        # Elongation Deltas
        baseline_elong = self.thresholds.get("baseline_values", {}).get("baseline_elongation", 280.0)
        features["elong_delta"] = features["ultrasonic_boot/elongation_avg"] - baseline_elong
        
        # Trends
        for sid, feat_prefix in [("ultrasonic_boot", "elong"), ("temperature_boot_material", "temp")]:
            for w in [60, 360, 1440]:
                w_name = f"{w//60}h" if w >= 60 else f"{w}m"
                # For streaming, we can't do perfect diff(60) without indexed state
                # We'll use (current - mean_of_buffer) or similar as a proxy if buffer is small
                # But here I'll try to get the actual oldest value if possible
                buffer = state.get("rolling_buffers", {}).get(sid, [])
                if len(buffer) >= w:
                    old_val = buffer[0] if len(buffer) == w else buffer[-w]
                    features[f"{feat_prefix}_trend_{w_name}"] = (val - old_val) / float(w)
                else:
                    features[f"{feat_prefix}_trend_{w_name}"] = 0.0

        # Degradation Index (Logic from fusion-model/ml_model/feature_engineering.py)
        delta_norm = np.clip(features["elong_delta"] / 30.0, 0.0, 2.0)
        trend_norm = np.clip(features.get("elong_trend_6h", 0.0) / 0.03, 0.0, 2.0)
        features["degradation_index"] = 0.55 * delta_norm + 0.30 * trend_norm
        features["degradation_rising_flag"] = int(features.get("elong_trend_6h", 0.0) > 0.01)

        # Thermal Features
        features["temp_above_baseline"] = max(features["temperature_boot_material/temperature_avg"] - 60.0, 0.0)
        features["temp_variability_12h"] = features.get("temperature_boot_material/temperature_12h_std", 0.0)
        features["temp_spike_index"] = features["temp_above_baseline"] / 20.0
        
        # Threshold Flags
        flags = self.derive_flags(event)
        features["temp_warning"] = flags.get("temperature_boot_material_warning", 0)
        features["temp_critical"] = flags.get("temperature_boot_material_critical", 0)
        features["elong_warning"] = flags.get("ultrasonic_boot_warning", 0)
        features["elong_critical"] = flags.get("ultrasonic_boot_critical", 0)
        features["current_warning"] = flags.get("current_transducer_warning", 0)
        features["current_critical"] = flags.get("current_transducer_critical", 0)
        
        features["warning_count"] = sum([features["temp_warning"], features["elong_warning"], features["current_warning"]])
        features["critical_count"] = sum([features["temp_critical"], features["elong_critical"], features["current_critical"]])

        # Fill missing with 0.0
        for f in self.required_features:
            if f not in features:
                features[f] = 0.0
                
        return features

    def get_ordered_vector(self, feature_dict: Dict[str, float]) -> np.ndarray:
        vector = [feature_dict.get(f, 0.0) for f in self.required_features]
        return np.array(vector).reshape(1, -1)
