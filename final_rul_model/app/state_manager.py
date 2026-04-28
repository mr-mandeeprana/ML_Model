"""
State Manager
Maintains rolling in-memory buffers for streaming feature generation.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional

logger = logging.getLogger(__name__)


class StateManager:
    def __init__(self, maxlen: int = 1440) -> None:
        """
        maxlen=1440 keeps up to 24 hours of 1-minute data.
        """
        self.maxlen = maxlen

        self.temperature: Deque[float] = deque(maxlen=maxlen)
        self.elongation: Deque[float] = deque(maxlen=maxlen)
        self.current: Deque[float] = deque(maxlen=maxlen)

        self.last_timestamp: Optional[str] = None
        self.last_sensorid: Optional[str] = None
        self.total_events: int = 0

    def _append_value(self, sensorid: str, value: float) -> None:
        if sensorid == "temperature_boot_material/temperature":
            self.temperature.append(value)
        elif sensorid == "ultrasonic_boot/elongation":
            self.elongation.append(value)
        elif sensorid == "current_transducer_head/current":
            self.current.append(value)

    def _is_ready(self) -> bool:
        """
        Require enough history for stable feature generation.
        Minimum useful threshold: at least 30 values for each sensor.
        """
        return (
            len(self.temperature) >= 30
            and len(self.elongation) >= 30
            and len(self.current) >= 30
        )

    def update(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update rolling state with one incoming event.

        Expected event:
        {
            "sensorid": "...",
            "@timestamp": "...",
            "avg_value": ...
        }
        """
        sensorid = str(event.get("sensorid", "")).strip()
        timestamp = event.get("@timestamp")

        try:
            value = float(event.get("avg_value", 0.0))
        except Exception:
            value = 0.0

        self._append_value(sensorid, value)

        self.last_timestamp = timestamp
        self.last_sensorid = sensorid
        self.total_events += 1

        state = {
            "temperature": list(self.temperature),
            "elongation": list(self.elongation),
            "current": list(self.current),
            "last_timestamp": self.last_timestamp,
            "last_sensorid": self.last_sensorid,
            "total_events": self.total_events,
            "ready": self._is_ready(),
        }

        return state