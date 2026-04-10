"""
Feature Engineering for Belt Health Prediction
Creates features from core sensors and synthetic targets for ML prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, 
                 belts_metadata_path: str = "config/belts_metadata.json",
                 thresholds_path: str = "config/thresholds.json"):
        
        self.belts_metadata_path = Path(belts_metadata_path)
        self.thresholds_path = Path(thresholds_path)
        
        self.metadata = self._load_json(self.belts_metadata_path)
        self.thresholds = self._load_json(self.thresholds_path)
        
    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            logger.warning(f"Config file not found at {path}. Using empty dict.")
            return {}
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}
    
    def pivot_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] > 10 and 'sensorid' not in df.columns:
            return df
        
        df['@timestamp'] = pd.to_datetime(df['@timestamp'])
        
        wide_df = df.pivot_table(
            index='@timestamp',
            columns='sensorid',
            values='avg_value',
            aggfunc='first'
        )
        
        wide_df.columns = [f"{col}_avg" for col in wide_df.columns]
        wide_df = wide_df.reset_index().sort_values('@timestamp')
        
        for col in wide_df.columns:
            if col != '@timestamp':
                wide_df[col] = wide_df[col].ffill(limit=2)
        
        return wide_df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('@timestamp').sort_index()
        windows = {'1h': 60, '12h': 720}
        
        core_cols = ['temperature_boot_material/temperature_avg',
                     'ultrasonic_boot/elongation_avg',
                     'current_transducer_head/current_avg']
        cols_to_roll = [c for c in core_cols if c in df.columns]
        
        for window_name, window_size in windows.items():
            for col in cols_to_roll:
                base = col.replace('_avg', '')
                df[f"{base}_{window_name}_mean"] = df[col].rolling(window=window_size, min_periods=1).mean()
                df[f"{base}_{window_name}_std"] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)
                df[f"{base}_{window_name}_min"] = df[col].rolling(window=window_size, min_periods=1).min()
                df[f"{base}_{window_name}_max"] = df[col].rolling(window=window_size, min_periods=1).max()
        
        return df.reset_index()
    
    def create_operational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        current_col = 'current_transducer_head/current_avg'
        if current_col not in df.columns:
            return df
        
        idle_threshold = self.thresholds.get('idle_detection', {}).get('current_threshold', 4.5)
        df['is_idle'] = (df[current_col] < idle_threshold).astype(int)
        df['operating_hours'] = (~df['is_idle'].astype(bool)).astype(int).cumsum() / 60.0
        
        high_load_threshold = self.thresholds.get('operational_states', {}).get('high_load_threshold', 62.0)
        df['high_load'] = (df[current_col] >= high_load_threshold).astype(int)
        
        df['current_percentile'] = df[current_col].rolling(window=720, min_periods=10).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-6) * 100, raw=False
        ).fillna(0)
        
        return df
    
    def create_degradation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        elong_col = 'ultrasonic_boot/elongation_avg'
        temp_col = 'temperature_boot_material/temperature_avg'
        current_col = 'current_transducer_head/current_avg'
        
        if elong_col in df.columns:
            df['elong_rate_12h'] = df[elong_col].diff(720).fillna(0)
            baseline_elong = self.thresholds.get('baseline_values', {}).get('baseline_elongation', 280.0)
            df['elong_delta'] = df[elong_col] - baseline_elong
            df['elong_delta_rate'] = df['elong_delta'].diff(60).fillna(0)

            # Degradation pattern features: trend, acceleration, and volatility.
            # Data is sampled at 1-minute intervals before the final downsample.
            df['elong_trend_1h'] = df[elong_col].diff(60).fillna(0) / 60.0
            df['elong_trend_6h'] = df[elong_col].diff(360).fillna(0) / 360.0
            df['elong_trend_24h'] = df[elong_col].diff(1440).fillna(0) / 1440.0
            df['elong_accel_6h'] = (df['elong_trend_1h'] - df['elong_trend_6h']).fillna(0)
            df['elong_volatility_6h'] = df[elong_col].rolling(window=360, min_periods=10).std().fillna(0)

            critical_elong = self.thresholds.get('sensor_warning_critical', {}).get('ultrasonic_boot/elongation', {}).get('critical', 310.0)
            elong_span = max(float(critical_elong - baseline_elong), 1.0)
            delta_norm = np.clip(df['elong_delta'] / elong_span, 0.0, 2.0)
            trend_norm = np.clip(df['elong_trend_6h'] / 0.03, 0.0, 2.0)
            vol_norm = np.clip(df['elong_volatility_6h'] / 6.0, 0.0, 2.0)

            # A compact scalar that summarizes current degradation severity.
            df['degradation_index'] = np.clip(
                0.55 * delta_norm + 0.30 * trend_norm + 0.15 * vol_norm,
                0.0,
                2.0
            )
            df['degradation_rising_flag'] = ((df['elong_trend_6h'] > 0.01) | (df['elong_accel_6h'] > 0.005)).astype(int)
        
        if temp_col in df.columns:
            baseline_temp = 60.0
            df['temp_above_baseline'] = np.maximum(df[temp_col] - baseline_temp, 0)
            df['temp_variability_12h'] = df[temp_col].rolling(window=720, min_periods=1).std().fillna(0)
            df['temp_trend_1h'] = df[temp_col].diff(60).fillna(0) / 60.0
            df['temp_trend_6h'] = df[temp_col].diff(360).fillna(0) / 360.0
            df['temp_trend_24h'] = df[temp_col].diff(1440).fillna(0) / 1440.0
            df['temp_spike_index'] = np.clip(
                (df[temp_col] - baseline_temp) / 20.0 + (df['temp_variability_12h'] / 8.0),
                0.0,
                3.0
            )

            if elong_col in df.columns and 'elong_trend_6h' in df.columns:
                # Explicit thermal-mechanical coupling: high temp with positive elongation trend.
                df['temp_elong_coupling'] = np.clip(
                    (df['temp_above_baseline'] / 25.0) * np.clip(df['elong_trend_6h'] / 0.03, 0.0, 2.0),
                    0.0,
                    3.0
                )
                df['temp_elong_interaction'] = df['temp_above_baseline'] * np.maximum(df['elong_delta'], 0.0)
                df['temp_elong_risk_flag'] = ((df['temp_above_baseline'] >= 15.0) & (df['elong_trend_6h'] > 0.01)).astype(int)

        if current_col in df.columns:
            df['current_volatility_6h'] = df[current_col].rolling(window=360, min_periods=10).std().fillna(0)
        
        return df
    
    def create_threshold_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        sensor_cfg = self.thresholds.get('sensor_warning_critical', {})
        
        if 'temperature_boot_material/temperature_avg' in df.columns:
            tcfg = sensor_cfg.get('temperature_boot_material/temperature', {})
            df['temp_warning'] = (df['temperature_boot_material/temperature_avg'] >= tcfg.get('warning', 90.0)).astype(int)
            df['temp_critical'] = (df['temperature_boot_material/temperature_avg'] >= tcfg.get('critical', 100.0)).astype(int)
        
        if 'ultrasonic_boot/elongation_avg' in df.columns:
            ecfg = sensor_cfg.get('ultrasonic_boot/elongation', {})
            df['elong_warning'] = (df['ultrasonic_boot/elongation_avg'] >= ecfg.get('warning', 300.0)).astype(int)
            df['elong_critical'] = (df['ultrasonic_boot/elongation_avg'] >= ecfg.get('critical', 310.0)).astype(int)
        
        if 'current_transducer_head/current_avg' in df.columns:
            ccfg = sensor_cfg.get('current_transducer_head/current', {})
            df['current_warning'] = (df['current_transducer_head/current_avg'] >= ccfg.get('warning', 62.0)).astype(int)
            df['current_critical'] = (df['current_transducer_head/current_avg'] >= ccfg.get('critical', 70.0)).astype(int)
        
        warn_cols = [c for c in df.columns if 'warning' in c and c != 'temp_warning']
        crit_cols = [c for c in df.columns if 'critical' in c]
        
        if warn_cols:
            df['warning_count'] = df[warn_cols].sum(axis=1)
        if crit_cols:
            df['critical_count'] = df[crit_cols].sum(axis=1)
        
        return df
    
    def create_synthetic_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        health = pd.Series(100.0, index=df.index)
        rul = pd.Series(2190.0, index=df.index)
        
        if 'temp_above_baseline' in df.columns:
            health -= np.clip(df['temp_above_baseline'] * 0.5, 0.0, 20.0)
            rul -= np.clip(df['temp_above_baseline'] * 10.0, 0.0, 400.0)
        
        if 'elong_delta' in df.columns:
            baseline_elong = self.thresholds.get('baseline_values', {}).get('baseline_elongation', 280.0)
            critical_elong = self.thresholds.get('sensor_warning_critical', {}).get('ultrasonic_boot/elongation', {}).get('critical', 310.0)
            elong_range = critical_elong - baseline_elong
            if elong_range > 0:
                health -= np.clip(df['elong_delta'] / elong_range * 30.0, 0.0, 30.0)
                rul -= np.clip(df['elong_delta'] / elong_range * 600.0, 0.0, 600.0)
        
        if 'current_transducer_head/current_avg' in df.columns:
            hlt = self.thresholds.get('operational_states', {}).get('high_load_threshold', 62.0)
            curr = df['current_transducer_head/current_avg'].fillna(0)
            health -= np.clip((curr - hlt) / 8.0 * 20.0, 0.0, 20.0)
            rul -= np.clip((curr - hlt) / 8.0 * 400.0, 0.0, 400.0)
        
        if 'temp_variability_12h' in df.columns:
            health -= np.clip(df['temp_variability_12h'] / 10.0 * 5.0, 0.0, 5.0)
            rul -= np.clip(df['temp_variability_12h'] / 10.0 * 100.0, 0.0, 100.0)

        if 'temp_spike_index' in df.columns:
            health -= np.clip(df['temp_spike_index'] * 2.5, 0.0, 7.5)
            rul -= np.clip(df['temp_spike_index'] * 55.0, 0.0, 150.0)

        if 'degradation_index' in df.columns:
            # Penalize targets when degradation pattern indicates sustained/rising wear.
            health -= np.clip(df['degradation_index'] * 12.0, 0.0, 20.0)
            rul -= np.clip(df['degradation_index'] * 280.0, 0.0, 500.0)

        if 'degradation_rising_flag' in df.columns:
            health -= df['degradation_rising_flag'] * 1.5
            rul -= df['degradation_rising_flag'] * 35.0

        if 'temp_elong_coupling' in df.columns:
            health -= np.clip(df['temp_elong_coupling'] * 3.5, 0.0, 10.0)
            rul -= np.clip(df['temp_elong_coupling'] * 90.0, 0.0, 220.0)

        if 'temp_elong_risk_flag' in df.columns:
            health -= df['temp_elong_risk_flag'] * 2.0
            rul -= df['temp_elong_risk_flag'] * 45.0
            
        df['target_health'] = np.clip(health, 0.0, 100.0)
        df['target_rul_days'] = np.clip(rul, 0.0, 5000.0)
        return df
    
    def run(self, input_path: str) -> pd.DataFrame:
        logger.info("Starting Feature Engineering...")
        if not Path(input_path).exists():
            raise FileNotFoundError(f"{input_path} not found.")
        
        # Taking every 5th row to speed up pipeline significantly for demonstration/testing
        df = pd.read_csv(input_path)
        df = df.iloc[::5, :]
        
        df_wide = self.pivot_to_wide(df)
        df_feats = self.create_rolling_features(df_wide)
        df_feats = self.create_operational_features(df_feats)
        df_feats = self.create_degradation_features(df_feats)
        df_feats = self.create_threshold_indicator_features(df_feats)
        df_feats = self.create_synthetic_targets(df_feats)
        return df_feats
