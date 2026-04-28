import json
import time
import os
import argparse
import random
import logging
from datetime import datetime, timezone
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IoTGateway")

DEFAULT_TOPIC = "belt-data"

class IoTGateway:
    def __init__(self, bootstrap_servers: str):
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                api_version=(2, 5, 0),
                acks='all',
                retries=5,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            logger.info(f"Connected to Kafka at {bootstrap_servers}")
        except Exception as e:
            logger.error(f"Failed to connect to Kafka at {bootstrap_servers}: {e}")
            self.producer = None

        # Data Generation definitions
        self.sensors = [
            {"id": "temperature_boot_material/temperature", "min": 50.0, "max": 120.0},
            {"id": "ultrasonic_boot/elongation", "min": 40.0, "max": 340.0},
            {"id": "current_transducer_head/current", "min": 0.0, "max": 70.0}
        ]
        
    def _generate_sensor_reading(self, sensor_cfg: dict) -> dict:
        """Generate a random reading for a specific sensor"""
        value = random.uniform(sensor_cfg["min"], sensor_cfg["max"])
        std_dev = random.uniform(0.1, 1.5)
        return {
            "belt_id": "belt-1",
            "sensorid": sensor_cfg["id"],
            "@timestamp": datetime.now(timezone.utc).isoformat(),
            "avg_value": value,
            "min_value": value - std_dev,
            "max_value": value + std_dev,
            "std_deviation": std_dev,
            "variance": std_dev ** 2,
            "median": value
        }

    def generate_and_stream(self, interval: float, topic: str = DEFAULT_TOPIC):
        """Continuously generate and stream live data infinitely"""
        if not self.producer:
            return

        logger.info(f"Starting to generate LIVE data natively to topic '{topic}' every {interval}s")
        count = 0
        
        try:
            while True:
                # Pick a random sensor to report from our active list
                sensor = random.choice(self.sensors)
                row = self._generate_sensor_reading(sensor)
                
                # Send to Kafka
                self.producer.send(topic, value=row)
                
                if count % 10 == 0:
                    logger.info(f"Generated: {row['sensorid']} -> Avg: {row['avg_value']:.2f}")
                
                count += 1
                if interval > 0:
                    time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Live data generation stopped by user.")
        finally:
            self.producer.flush()
            logger.info(f"Stream complete. Sent {count} records.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Gateway Live Data Generator")
    parser.add_argument("--simulate", action="store_true", help="Run simulation")
    parser.add_argument("--interval", type=float, default=2.0, help="Interval between sends in seconds")
    
    args = parser.parse_args()
    
    kafka_broker = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    gateway = IoTGateway(kafka_broker)
    
    if args.simulate:
        gateway.generate_and_stream(args.interval)
