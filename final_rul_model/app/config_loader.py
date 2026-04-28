"""
Config Loader
Loads JSON configuration files for runtime usage.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent

        self.thresholds_path = Path(
            os.getenv("THRESHOLDS_PATH", str(base_dir / "config" / "thresholds.json"))
        )
        self.model_config_path = Path(
            os.getenv("MODEL_CONFIG_PATH", str(base_dir / "config" / "model_config.json"))
        )
        self.metadata_path = Path(
            os.getenv("BELT_METADATA_PATH", str(base_dir / "config" / "belts_metadata.json"))
        )

        self._thresholds = self._load_json(self.thresholds_path)
        self._model_config = self._load_json(self.model_config_path)
        self._metadata = self._load_json(self.metadata_path)

        logger.info("ConfigLoader initialized.")

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            return {}

    def get_thresholds(self) -> Dict[str, Any]:
        return self._thresholds

    def get_model_config(self) -> Dict[str, Any]:
        return self._model_config

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata