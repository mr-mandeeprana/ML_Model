# scripts/delta_catchup.py

import os
import json
import time
from datetime import datetime
from elasticsearch import Elasticsearch
from kafka import KafkaProducer

# --- CONFIGURATION ---
ELASTIC_URL = os.getenv("ELASTIC_URL", "http://localhost:9200")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")

STATE_INDEX = "belt-runtime-state"
SOURCE_INDEX = "belt-raw-data"  # The index where raw sensor data is archived
TARGET_TOPIC = "belt-data"     # The input topic for the Numaflow pipeline

def get_latest_states(es):
    """Fetch the last known state for all belts."""
    print(f"🔍 Fetching anchor states from {STATE_INDEX}...")
    try:
        query = {"query": {"match_all": {}}}
        resp = es.search(index=STATE_INDEX, body=query, size=100)
        return [hit["_source"] for hit in resp["hits"]["hits"]]
    except Exception as e:
        print(f"❌ Error fetching states: {e}")
        return []

def catch_up_belt(es, producer, belt_state):
    belt_id = belt_state.get("belt_id")
    last_ts = belt_state.get("last_prediction_timestamp")
    
    if not belt_id or not last_ts:
        return

    print(f"📈 Processing catch-up for Belt: {belt_id} (Last Anchor: {last_ts})")

    # Query source index for missing data
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"belt_id": belt_id}},
                    {"range": {"timestamp": {"gt": last_ts}}}
                ]
            }
        },
        "sort": [{"timestamp": "asc"}]
    }

    try:
        # Use scroll if data might be large, but for catch-up periodic runs search hits are usually enough
        resp = es.search(index=SOURCE_INDEX, body=query, size=1000)
        hits = resp["hits"]["hits"]
        count = len(hits)
        
        if count == 0:
            print(f"✅ Belt {belt_id} is already up to date.")
            return

        print(f"🚀 Replaying {count} events to Kafka topic '{TARGET_TOPIC}'...")
        
        for hit in hits:
            event = hit["_source"]
            producer.send(TARGET_TOPIC, value=event)
        
        producer.flush()
        print(f"🎯 Successfully re-injected {count} records for {belt_id}.")

    except Exception as e:
        print(f"❌ Error during catch-up for {belt_id}: {e}")

def main():
    print("=========================================")
    print("   BELT ML RUNTIME - DELTA CATCH-UP     ")
    print("=========================================")

    es = Elasticsearch(ELASTIC_URL)
    # Check if we need to force compatibility (though v8 client should do this by default)
    # es = Elasticsearch(ELASTIC_URL, headers={"Accept": "application/vnd.elasticsearch+json; compatible-with=8"})

    
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    except Exception as e:
        print(f"❌ Kafka Connection Failed: {e}")
        return

    states = get_latest_states(es)
    if not states:
        print("⚠ No belt states found to process.")
        return

    for state in states:
        catch_up_belt(es, producer, state)

    print("=========================================")
    print("   CATCH-UP SEQUENCE COMPLETE           ")
    print("=========================================")

if __name__ == "__main__":
    main()
