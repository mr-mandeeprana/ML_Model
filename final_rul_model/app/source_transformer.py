"""
Source Transformer
Transforms raw Kafka sensor events into clean normalized format
for downstream Numaflow processing.
"""

import json
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()


def process_event(event: dict) -> dict:
    """
    Expected input:
    {
        "belt_id": "...",
        "sensorid": "...",
        "@timestamp": "...",
        "avg_value": ...
    }
    """

    try:
        belt_id = str(event.get("belt_id", "belt-1")).strip() or "belt-1"
        sensor = str(event.get("sensorid", "")).strip()
        timestamp = normalize_timestamp(str(event.get("@timestamp", "")))
        value = float(event.get("avg_value", 0.0))

        std_dev = float(event.get("std_deviation", 0.0))
        min_value = float(event.get("min_value", value - std_dev))
        max_value = float(event.get("max_value", value + std_dev))
        variance = float(event.get("variance", std_dev ** 2))
        median = float(event.get("median", value))

        return {
            "belt_id": belt_id,
            "sensorid": sensor,
            "@timestamp": timestamp,
            "avg_value": value,
            "min_value": min_value,
            "max_value": max_value,
            "std_deviation": std_dev,
            "variance": variance,
            "median": median,
        }

    except Exception as e:
        logger.error(f"Error processing event: {e}")
        return None


def main():
    """
    Reads from stdin (Numaflow input)
    Writes to stdout (Numaflow output)
    """
    import sys

    for line in sys.stdin:
        try:
            event = json.loads(line.strip())
            processed = process_event(event)

            if processed:
                print(json.dumps(processed), flush=True)

        except Exception as e:
            logger.error(f"Invalid input: {e}")


if __name__ == "__main__":
    main()