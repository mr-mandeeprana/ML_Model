# app/alert_engine.py

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AlertEngine:
    """
    Evaluates predictions against thresholds.
    Supports hot-reloading of thresholds.json.
    """
    def __init__(self, thresholds_path: str = "config/thresholds.json"):
        self._path = Path(thresholds_path)
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0
        self.thresholds: Dict[str, Any] = {}
        self._reload()

    def _reload(self) -> None:
        if not self._path.exists():
            logger.warning(f"Thresholds file not found at {self._path}")
            self.thresholds = {}
        else:
            try:
                with open(self._path, "r") as f:
                    self.thresholds = json.load(f)
                self._last_mtime = self._path.stat().st_mtime
            except Exception as e:
                logger.error(f"Failed to load thresholds: {e}")

        # Map thresholds from fusion-model config structure
        # (Assuming they exist or use defaults based on original alert_engine)
        hs = self.thresholds.get("health_score_thresholds", {})
        rul = self.thresholds.get("rul_thresholds", {})
        
        self.hs_critical = float(hs.get("critical", 55.0))
        self.hs_warning = float(hs.get("warning", 65.0))
        self.rul_critical = float(rul.get("critical_days", 180))
        self.rul_warning = float(rul.get("warning_days", 540))

    def evaluate(self, health: float, rul: float) -> Dict[str, Any]:
        # Hot-reload check every 60s
        now = time.monotonic()
        if now - self._last_check > 60:
            self._last_check = now
            try:
                if self._path.stat().st_mtime != self._last_mtime:
                    self._reload()
            except: pass

        is_critical = (health < self.hs_critical) or (rul < self.rul_critical)
        is_warning = (health < self.hs_warning) or (rul < self.rul_warning)

        if is_critical: risk = "CRITICAL"
        elif is_warning: risk = "WARNING"
        else: risk = "NORMAL"

        return {
            "is_critical": is_critical,
            "is_warning": is_warning and not is_critical,
            "risk_level": risk
        }
