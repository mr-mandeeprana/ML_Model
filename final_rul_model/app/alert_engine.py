"""
Alert Engine
Applies business thresholds to ML predictions and returns alert status.
"""

from __future__ import annotations

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AlertEngine:
    def __init__(self, config_loader) -> None:
        self.config_loader = config_loader
        self.thresholds = self.config_loader.get_thresholds()

        hs_cfg = self.thresholds.get("health_score_thresholds", {})
        rul_cfg = self.thresholds.get("rul_thresholds", {})

        self.health_critical = float(hs_cfg.get("critical", 40.0))
        self.health_warning = float(hs_cfg.get("warning", 55.0))
        self.health_maintenance_due = float(hs_cfg.get("maintenance_due", 65.0))
        self.health_good = float(hs_cfg.get("good", 75.0))
        self.health_excellent = float(hs_cfg.get("excellent", 90.0))

        self.rul_critical = float(rul_cfg.get("critical_days", 180.0))
        self.rul_warning = float(rul_cfg.get("warning_days", 540.0))
        self.rul_maintenance_due = float(rul_cfg.get("maintenance_due_days", 1080.0))

    def _classify_risk(self, health: float, rul_days: float) -> str:
        if health <= self.health_critical or rul_days <= self.rul_critical:
            return "CRITICAL"
        if health <= self.health_warning or rul_days <= self.rul_warning:
            return "WARNING"
        return "NORMAL"

    def _classify_health_band(self, health: float) -> str:
        if health >= self.health_excellent:
            return "EXCELLENT"
        if health >= self.health_good:
            return "GOOD"
        if health >= self.health_maintenance_due:
            return "MAINTENANCE_DUE"
        if health >= self.health_warning:
            return "WARNING"
        return "CRITICAL"

    def _classify_rul_band(self, rul_days: float) -> str:
        if rul_days <= self.rul_critical:
            return "CRITICAL"
        if rul_days <= self.rul_warning:
            return "WARNING"
        if rul_days <= self.rul_maintenance_due:
            return "MAINTENANCE_DUE"
        return "NORMAL"

    def apply(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:
            {
              "ml_health": float,
              "ml_rul_days": float
            }

        Output:
            {
              "ml_health": ...,
              "ml_rul_days": ...,
              "risk": ...,
              "health_band": ...,
              "rul_band": ...,
              "is_alert": ...,
              "is_critical": ...,
              "is_warning": ...
            }
        """
        health = float(prediction.get("ml_health", 90.0))
        rul_days = float(prediction.get("ml_rul_days", 2190.0))

        risk = self._classify_risk(health, rul_days)
        health_band = self._classify_health_band(health)
        rul_band = self._classify_rul_band(rul_days)

        result = {
            "ml_health": round(health, 2),
            "ml_rul_days": round(rul_days, 2),
            "risk": risk,
            "health_band": health_band,
            "rul_band": rul_band,
            "is_alert": risk in {"WARNING", "CRITICAL"},
            "is_critical": risk == "CRITICAL",
            "is_warning": risk == "WARNING",
        }

        return result