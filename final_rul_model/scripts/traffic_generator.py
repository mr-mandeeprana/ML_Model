import json
import time
import os
import argparse
from datetime import datetime, timezone, timedelta
from kafka import KafkaProducer

class TrafficGenerator:
    def __init__(self, kafka_bootstrap=None):
        self.bootstrap_servers = kafka_bootstrap or os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.topic = "belt-data"

    def replay_file(self, file_path, speed_up=1.0, retime=True):
        """
        Replays data from a JSONL file.
        'retime' adjusts timestamps to start 'now'.
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return

        print(f"Replaying {file_path} at {speed_up}x speed...")
        
        start_time_now = datetime.now(timezone.utc)
        first_event_ts = None
        
        with open(file_path, "r") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    ts_str = record.get("@timestamp") or record.get("timestamp")
                    if not ts_str:
                        continue
                        
                    event_ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    
                    if first_event_ts is None:
                        first_event_ts = event_ts
                        
                    # Calculate delay
                    elapsed_event = (event_ts - first_event_ts).total_seconds()
                    elapsed_now = (datetime.now(timezone.utc) - start_time_now).total_seconds()
                    
                    wait_time = (elapsed_event / speed_up) - elapsed_now
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # Retime
                    if retime:
                        record["@timestamp"] = (start_time_now + timedelta(seconds=elapsed_event)).isoformat()
                    
                    self.producer.send(self.topic, record)
                    print(f"Sent record for {record.get('sensorid')} @ {record.get('@timestamp')}")
                    
                except Exception as e:
                    print(f"Error processing line: {e}")

        self.producer.flush()
        print("Replay complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL log file")
    parser.add_argument("--speed", type=float, default=1.0, help="Replay speedup factor")
    parser.add_argument("--kafka", type=str, help="Kafka bootstrap")
    parser.add_argument("--no-retime", action="store_false", dest="retime", help="Don't adjust timestamps to now")
    
    args = parser.parse_args()
    
    gen = TrafficGenerator(kafka_bootstrap=args.kafka)
    gen.replay_file(args.file, speed_up=args.speed, retime=args.retime)
