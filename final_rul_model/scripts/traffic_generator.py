import json
import time
import random
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

sensors = [
    "temperature_boot_material/temperature",
    "ultrasonic_boot/elongation",
    "current_transducer_head/current",
]

while True:
    sensor = random.choice(sensors)

    if "temperature" in sensor:
        value = random.uniform(60, 100)
    elif "elongation" in sensor:
        value = random.uniform(180, 320)
    else:
        value = random.uniform(20, 70)

    data = {
        "sensorid": sensor,
        "@timestamp": datetime.utcnow().isoformat(),
        "avg_value": value,
    }

    producer.send("belt-data", data)
    print("Sent:", data)

    time.sleep(1)