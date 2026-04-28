import pandas as pd
import json
from kafka import KafkaProducer
import time

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

df = pd.read_csv("data.csv")

for _, row in df.iterrows():
    event = {
        "sensorid": row["sensorid"],
        "@timestamp": row["@timestamp"],
        "avg_value": row["avg_value"],
    }

    producer.send("belt-data", event)
    print("Replayed:", event)

    time.sleep(0.1)