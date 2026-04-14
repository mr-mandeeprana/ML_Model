# app/state_manager.py

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages persistent state for belts.
    Currently in-memory (local dict), but structured for Redis integration.
    """
    def __init__(self):
        # In-memory store: { belt_id: { state_data } }
        self._store: Dict[str, Dict[str, Any]] = {}
        
        # Max buffer size for 24h of data (sampled at 1 min)
        self.MAX_BUFFER_SIZE = 1440 

    def get_state(self, belt_id: str) -> Dict[str, Any]:
        if belt_id not in self._store:
            # Initialize default state for new belt
            self._store[belt_id] = {
                "belt_id": belt_id,
                "operating_hours": 0.0,
                "rolling_buffers": {}, # { sensor_id: deque([values]) }
                "last_prediction_timestamp": None,
                "health_score": 100.0,
                "rul_days": 2190.0
            }
        return self._store[belt_id]

    def save_state(self, belt_id: str, state: Dict[str, Any]):
        self._store[belt_id] = state

    def update_buffer(self, belt_id: str, sensor_id: str, value: float):
        state = self.get_state(belt_id)
        buffers = state.setdefault("rolling_buffers", {})
        
        if sensor_id not in buffers:
            buffers[sensor_id] = deque(maxlen=self.MAX_BUFFER_SIZE)
        
        # If it's a deque (deserialized from JSON or newly created), append
        if not isinstance(buffers[sensor_id], deque):
            buffers[sensor_id] = deque(buffers[sensor_id], maxlen=self.MAX_BUFFER_SIZE)
            
        buffers[sensor_id].append(value)

    def get_buffer_stats(self, belt_id: str, sensor_id: str, window_minutes: int) -> Dict[str, float]:
        state = self.get_state(belt_id)
        buffer = state.get("rolling_buffers", {}).get(sensor_id)
        
        if not buffer or len(buffer) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        
        # Take the last N elements
        data = list(buffer)[-window_minutes:]
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data))
        }

    def increment_operating_hours(self, belt_id: str, hours: float = 1/60.0):
        state = self.get_state(belt_id)
        state["operating_hours"] = state.get("operating_hours", 0.0) + hours
