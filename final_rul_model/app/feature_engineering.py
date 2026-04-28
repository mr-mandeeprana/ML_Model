"""
Streaming Feature Builder
Creates runtime features from state buffers (real-time).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class FeatureBuilder:
    def __init__(self, config_loader):
        self.config_loader = config_loader
        thresholds = config_loader.get_thresholds()

        sensor_cfg = thresholds.get("sensor_thresholds", {})
        baseline = thresholds.get("baseline_values", {})
        ops = thresholds.get("operational_states", {})
        idle = thresholds.get("idle_detection", {})

        self.temp_key = "temperature_boot_material/temperature"
        self.elong_key = "ultrasonic_boot/elongation"
        self.current_key = "current_transducer_head/current"

        tcfg = sensor_cfg.get(self.temp_key, {})
        self.temp_warn = float(tcfg.get("warning", 90.0))
        self.temp_critical = float(tcfg.get("critical", 100.0))

        ecfg = sensor_cfg.get(self.elong_key, {})
        self.elong_warn = float(ecfg.get("warning", 300.0))
        self.elong_critical = float(ecfg.get("critical", 310.0))

        ccfg = sensor_cfg.get(self.current_key, {})
        self.current_warn = float(ccfg.get("warning", 62.0))
        self.current_critical = float(ccfg.get("critical", 80.0))

        self.baseline_elong = float(baseline.get("baseline_elongation", 280.0))
        self.baseline_temp = float(baseline.get("baseline_temp_celsius", 60.0))

        self.idle_current_threshold = float(idle.get("current_threshold", 4.5))
        self.high_load_threshold = float(ops.get("high_load_threshold", 62.0))

    def _safe_array(self, data):
        if not data:
            return np.array([], dtype=float)
        return np.array(data, dtype=float)

    def _mean(self, arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0

    def _std(self, arr):
        return float(np.std(arr)) if len(arr) > 1 else 0.0

    def _min(self, arr):
        return float(np.min(arr)) if len(arr) > 0 else 0.0

    def _max(self, arr):
        return float(np.max(arr)) if len(arr) > 0 else 0.0

    def _trend(self, arr):
        if len(arr) < 2:
            return 0.0
        x = np.arange(len(arr))
        return float(np.polyfit(x, arr, 1)[0])

    def _rate(self, arr):
        if len(arr) < 2:
            return 0.0
        return float((arr[-1] - arr[0]) / max(len(arr) - 1, 1))

    def _accel(self, arr):
        if len(arr) < 3:
            return 0.0
        # Simple acceleration: mean of change in trend over 6h vs 1h
        return 0.0 # Placeholder for complex accel if needed

    def build(self, state):
        try:
            temp_arr = self._safe_array(state["temperature"])
            elong_arr = self._safe_array(state["elongation"])
            current_arr = self._safe_array(state["current"])

            if len(temp_arr) == 0 or len(elong_arr) == 0 or len(current_arr) == 0:
                return None

            temp = float(temp_arr[-1])
            elong = float(elong_arr[-1])
            current = float(current_arr[-1])

            # Buffers
            temp_1h = temp_arr[-60:]
            elong_1h = elong_arr[-60:]
            curr_1h = current_arr[-60:]

            temp_6h = temp_arr[-360:]
            elong_6h = elong_arr[-360:]
            curr_6h = current_arr[-360:]

            temp_12h = temp_arr[-720:]
            elong_12h = elong_arr[-720:]
            curr_12h = current_arr[-720:]

            temp_24h = temp_arr[-1440:]
            elong_24h = elong_arr[-1440:]
            curr_24h = current_arr[-1440:]

            features = {}

            # Base
            features["current_transducer_head/current_avg"] = current
            features["temperature_boot_material/temperature_avg"] = temp
            features["ultrasonic_boot/elongation_avg"] = elong

            # 1h stats
            features["temperature_boot_material/temperature_1h_mean"] = self._mean(temp_1h)
            features["temperature_boot_material/temperature_1h_std"] = self._std(temp_1h)
            features["temperature_boot_material/temperature_1h_min"] = self._min(temp_1h)
            features["temperature_boot_material/temperature_1h_max"] = self._max(temp_1h)

            features["ultrasonic_boot/elongation_1h_mean"] = self._mean(elong_1h)
            features["ultrasonic_boot/elongation_1h_std"] = self._std(elong_1h)
            features["ultrasonic_boot/elongation_1h_min"] = self._min(elong_1h)
            features["ultrasonic_boot/elongation_1h_max"] = self._max(elong_1h)

            features["current_transducer_head/current_1h_mean"] = self._mean(curr_1h)
            features["current_transducer_head/current_1h_std"] = self._std(curr_1h)
            features["current_transducer_head/current_1h_min"] = self._min(curr_1h)
            features["current_transducer_head/current_1h_max"] = self._max(curr_1h)

            # 6h stats
            features["temperature_boot_material/temperature_6h_mean"] = self._mean(temp_6h)
            features["temperature_boot_material/temperature_6h_std"] = self._std(temp_6h)
            features["temperature_boot_material/temperature_6h_min"] = self._min(temp_6h)
            features["temperature_boot_material/temperature_6h_max"] = self._max(temp_6h)

            features["ultrasonic_boot/elongation_6h_mean"] = self._mean(elong_6h)
            features["ultrasonic_boot/elongation_6h_std"] = self._std(elong_6h)
            features["ultrasonic_boot/elongation_6h_min"] = self._min(elong_6h)
            features["ultrasonic_boot/elongation_6h_max"] = self._max(elong_6h)

            features["current_transducer_head/current_6h_mean"] = self._mean(curr_6h)
            features["current_transducer_head/current_6h_std"] = self._std(curr_6h)
            features["current_transducer_head/current_6h_min"] = self._min(curr_6h)
            features["current_transducer_head/current_6h_max"] = self._max(curr_6h)

            # 12h stats
            features["temperature_boot_material/temperature_12h_mean"] = self._mean(temp_12h)
            features["temperature_boot_material/temperature_12h_std"] = self._std(temp_12h)
            features["temperature_boot_material/temperature_12h_min"] = self._min(temp_12h)
            features["temperature_boot_material/temperature_12h_max"] = self._max(temp_12h)

            features["ultrasonic_boot/elongation_12h_mean"] = self._mean(elong_12h)
            features["ultrasonic_boot/elongation_12h_std"] = self._std(elong_12h)
            features["ultrasonic_boot/elongation_12h_min"] = self._min(elong_12h)
            features["ultrasonic_boot/elongation_12h_max"] = self._max(elong_12h)

            features["current_transducer_head/current_12h_mean"] = self._mean(curr_12h)
            features["current_transducer_head/current_12h_std"] = self._std(curr_12h)
            features["current_transducer_head/current_12h_min"] = self._min(curr_12h)
            features["current_transducer_head/current_12h_max"] = self._max(curr_12h)

            # 24h stats
            features["temperature_boot_material/temperature_24h_mean"] = self._mean(temp_24h)
            features["temperature_boot_material/temperature_24h_std"] = self._std(temp_24h)
            features["temperature_boot_material/temperature_24h_min"] = self._min(temp_24h)
            features["temperature_boot_material/temperature_24h_max"] = self._max(temp_24h)

            features["ultrasonic_boot/elongation_24h_mean"] = self._mean(elong_24h)
            features["ultrasonic_boot/elongation_24h_std"] = self._std(elong_24h)
            features["ultrasonic_boot/elongation_24h_min"] = self._min(elong_24h)
            features["ultrasonic_boot/elongation_24h_max"] = self._max(elong_24h)

            features["current_transducer_head/current_24h_mean"] = self._mean(curr_24h)
            features["current_transducer_head/current_24h_std"] = self._std(curr_24h)
            features["current_transducer_head/current_24h_min"] = self._min(curr_24h)
            features["current_transducer_head/current_24h_max"] = self._max(curr_24h)

            # Ops/state
            features["is_idle"] = int(current < self.idle_current_threshold)
            features["high_load"] = int(current >= self.high_load_threshold)
            
            active_flags = (curr_1h >= self.idle_current_threshold).astype(int)
            features["utilization_1h"] = float(np.mean(active_flags)) if len(active_flags) > 0 else 0.0
            
            active_flags_6h = (curr_6h >= self.idle_current_threshold).astype(int)
            features["utilization_6h"] = float(np.mean(active_flags_6h)) if len(active_flags_6h) > 0 else 0.0
            
            high_load_flags_6h = (curr_6h >= self.high_load_threshold).astype(int)
            features["high_load_share_6h"] = float(np.mean(high_load_flags_6h)) if len(high_load_flags_6h) > 0 else 0.0

            # Percentile
            curr_720 = current_arr[-720:]
            curr_span = self._max(curr_720) - self._min(curr_720)
            features["current_percentile"] = (
                float((current - self._min(curr_720)) / (curr_span + 1e-6) * 100.0)
                if len(curr_720) > 0
                else 0.0
            )

            # Elongation degradation
            features["elong_rate_12h"] = float(elong - elong_arr[-720]) if len(elong_arr) >= 720 else 0.0
            features["elong_delta"] = float(elong - self.baseline_elong)
            features["elong_delta_rate"] = float(elong - elong_arr[-60]) if len(elong_arr) >= 60 else 0.0
            features["elong_trend_1h"] = self._trend(elong_1h)
            features["elong_trend_6h"] = self._trend(elong_6h)
            features["elong_trend_24h"] = self._trend(elong_24h)
            features["elong_micro_trend"] = float(elong - elong_arr[-5]) if len(elong_arr) >= 5 else 0.0
            features["elong_accel_6h"] = float(features["elong_trend_1h"] - features["elong_trend_6h"])
            features["elong_volatility_6h"] = self._std(elong_6h)
            
            # Degradation Index (Same formula as source)
            elong_span = 30.0 # critical (310) - baseline (280)
            delta_norm = np.clip(features["elong_delta"] / elong_span, 0.0, 2.0)
            trend_norm = np.clip(features["elong_trend_6h"] / 0.03, 0.0, 2.0)
            vol_norm = np.clip(features["elong_volatility_6h"] / 6.0, 0.0, 2.0)
            features["degradation_index"] = float(np.clip(
                0.55 * delta_norm + 0.30 * trend_norm + 0.15 * vol_norm,
                0.0,
                2.0,
            ))
            features["degradation_rising_flag"] = int(
                features["elong_trend_6h"] > 0.01 or features["elong_accel_6h"] > 0.005
            )

            # Current/temp
            features["current_volatility_6h"] = self._std(curr_6h)
            features["temp_above_baseline"] = float(max(0.0, temp - self.baseline_temp))
            features["temp_variability_12h"] = self._std(temp_12h)
            features["temp_trend_1h"] = self._trend(temp_1h)
            features["temp_trend_6h"] = self._trend(temp_6h)
            features["temp_trend_24h"] = self._trend(temp_24h)
            features["temp_spike_index"] = float(np.clip(
                (features["temp_above_baseline"] / 20.0) + (features["temp_variability_12h"] / 8.0),
                0.0,
                3.0,
            ))

            # Interaction
            features["temp_elong_coupling"] = float(np.clip(
                (features["temp_above_baseline"] / 25.0) * trend_norm,
                0.0,
                3.0,
            ))
            features["temp_elong_interaction"] = float(features["temp_above_baseline"] * max(features["elong_delta"], 0.0))
            features["temp_elong_risk_flag"] = int(
                features["temp_above_baseline"] >= 15.0 and features["elong_trend_6h"] > 0.01
            )

            # Thresholds
            features["temp_warning"] = int(temp >= self.temp_warn)
            features["temp_critical"] = int(temp >= self.temp_critical)
            features["elong_warning"] = int(elong >= self.elong_warn)
            features["elong_critical"] = int(elong >= self.elong_critical)
            features["current_warning"] = int(current >= self.current_warn)
            features["current_critical"] = int(current >= self.current_critical)

            features["warning_count"] = int(
                features["temp_warning"] + features["elong_warning"] + features["current_warning"]
            )
            features["critical_count"] = int(
                features["temp_critical"] + features["elong_critical"] + features["current_critical"]
            )
            
            # Proximity and Trigger
            features["failure_proximity"] = float(np.clip(
                features["degradation_index"] + 0.5 * features["critical_count"] + 0.2 * features["warning_count"],
                0.0,
                5.0,
            ))
            features["failure_trigger"] = int(
                features["critical_count"] >= 2 and features["degradation_index"] >= 0.8
            )

            return features

        except Exception as e:
            logger.exception("Feature build failed: %s", e)
            return None