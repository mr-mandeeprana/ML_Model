# app/standalone_runtime.py

import json
import os
import logging
import time
from kafka import KafkaConsumer, KafkaProducer
from app.runtime import RuntimeEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Environment variables (overridable by docker-compose)
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    input_topic = os.getenv("KAFKA_INPUT_TOPIC", "belt-data")
    output_topic = os.getenv("KAFKA_OUTPUT_TOPIC", "belt-predictions")
    group_id = os.getenv("KAFKA_GROUP_ID", "belt-ml-standalone-group")

    logger.info(f"Initializing Standalone ML Runtime...")
    logger.info(f"Connecting to Kafka at {bootstrap_servers}")

    # Initialize ML Runtime (Loads models into memory ~1GB)
    try:
        runtime = RuntimeEngine()
        logger.info("ML Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        return

    # Initialize Kafka Consumer
    consumer = None
    while consumer is None:
        try:
            consumer = KafkaConsumer(
                input_topic,
                bootstrap_servers=bootstrap_servers,
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                group_id=group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
        except Exception as e:
            logger.warning(f"Could not connect to Kafka yet: {e}. Retrying in 5s...")
            time.sleep(5)

    # Initialize Kafka Producer
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        acks="all",
        retries=5,
        value_serializer=lambda x: json.dumps(x).encode('utf-8')
    )

    logger.info(f"Started consuming from {input_topic}...")

    for message in consumer:
        try:
            raw_event = message.value
            logger.debug(f"Received event from belt: {raw_event.get('belt_id')}")

            # Process using ML Runtime
            result = runtime.process_event(raw_event)
            if result is None:
                continue

            # Replicate key-based partitioning (use belt_id as key)
            key = raw_event.get("belt_id", "GLOBAL").encode('utf-8')

            # Produce to output topic
            producer.send(output_topic, value=result, key=key)
            
            # Since kafka-python is async by default, we can flush periodically 
            # or just let it handle it. For low latency, we don't need a flush here.
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")

if __name__ == "__main__":
    main()
