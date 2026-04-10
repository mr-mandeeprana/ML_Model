# app/config_loader.py

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)

    def load_json(self, filename: str) -> Dict[str, Any]:
        path = self.config_dir / filename
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return {}

    def get_thresholds(self) -> Dict[str, Any]:
        return self.load_json("thresholds.json")

    def get_model_config(self) -> Dict[str, Any]:
        return self.load_json("model_config.json")

    def get_metadata(self) -> Dict[str, Any]:
        return self.load_json("belts_metadata.json")
