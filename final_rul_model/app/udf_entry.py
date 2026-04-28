"""
Numaflow UDF Entry
Connects incoming stream events to runtime processing logic.
"""

import json
import logging
import sys

from app.runtime import RuntimeEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize runtime once (important for performance)
runtime = RuntimeEngine()


def process_line(line: str):
    try:
        event = json.loads(line.strip())

        # Process through runtime pipeline
        result = runtime.process_event(event)

        if result:
            print(json.dumps(result), flush=True)

    except Exception as e:
        logger.error(f"UDF processing error: {e}")


def main():
    """
    Numaflow reads stdin and expects output on stdout
    """
    for line in sys.stdin:
        process_line(line)


if __name__ == "__main__":
    main()