"""
Data Preprocessing Module
Loading, cleaning, and validating sensor data for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_path: str = "config/thresholds.json"):
        self.config_path = Path(config_path)
        self.thresholds = self._load_config()

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            logger.warning(f"Config not found at {self.config_path}. Using empty defaults.")
            return {}
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def load_sensor_data(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV: {e}")

        # Keep only the 3 core sensors
        core_sensors = [
            'temperature_boot_material/temperature',
            'ultrasonic_boot/elongation',
            'current_transducer_head/current'
        ]
        df = df[df['sensorid'].isin(core_sensors)].copy()
        
        if df.empty:
            raise ValueError("No core sensor data found. Required sensors: " + ", ".join(core_sensors))

        required_cols = ['sensorid', '@timestamp', 'avg_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
             raise ValueError(f"Missing critical columns: {missing_cols}")
             
        df['@timestamp'] = pd.to_datetime(df['@timestamp'], errors='coerce')
        df = df.dropna(subset=['@timestamp'])

        df['avg_value'] = pd.to_numeric(df['avg_value'], errors='coerce')
        
        for sensor in core_sensors:
            mask = df['sensorid'] == sensor
            if mask.any():
                df.loc[mask, 'avg_value'] = df.loc[mask, 'avg_value'].fillna(
                    df.loc[mask, 'avg_value'].median()
                )

        return df.sort_values(['@timestamp', 'sensorid'])

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=['@timestamp', 'sensorid'])
        
        sensor_limits = self.thresholds.get('sensor_limits', {})
        
        for sensor_key, limits in sensor_limits.items():
            min_val = limits.get('min', -np.inf)
            max_val = limits.get('max', np.inf)
            
            mask = df['sensorid'].str.contains(sensor_key.split('/')[0], regex=False)
            
            if mask.any():
                valid_value_mask = (df['avg_value'] >= min_val) & (df['avg_value'] <= max_val)
                df = df[~mask | valid_value_mask]
        
        logger.info(f"Cleaned {len(df)} rows remaining (removed out-of-range values)")
        return df

    def fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(['sensorid', '@timestamp'])
        
        cols_to_fill = ['avg_value']
        df[cols_to_fill] = df.groupby('sensorid')[cols_to_fill].ffill()
        df[cols_to_fill] = df.groupby('sensorid')[cols_to_fill].bfill()
        
        for sensor in df['sensorid'].unique():
            mask = df['sensorid'] == sensor
            sensor_median = df.loc[mask, 'avg_value'].median()
            df.loc[mask & df['avg_value'].isna(), 'avg_value'] = sensor_median
        
        return df.sort_values(['@timestamp', 'sensorid'])

    def preprocess(self, file_path: str) -> pd.DataFrame:
        logger.info("Starting preprocessing...")
        df = self.load_sensor_data(file_path)
        logger.info(f"Loaded {len(df)} rows from {len(df['sensorid'].unique())} sensors.")
        
        df = self.clean_data(df)
        df = self.fill_missing_values(df)
        
        logger.info("Preprocessing complete.")
        return df
