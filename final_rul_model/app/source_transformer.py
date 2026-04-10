import sys
import json
import grpc
from datetime import datetime
from pynumaflow.sourcetransformer import Messages, Message, Datum, SourceTransformServer

def transform_handler(keys: list[str], datum: Datum) -> Messages:
    """
    Handles event time extraction and belt_id injection.
    Expects input: {sensorid, @timestamp, max_value, min_value, avg_value, std_deviation, belt_id}
    """
    try:
        payload = json.loads(datum.value.decode("utf-8"))
        
        # 1. Extract Event Time
        ts_str = payload.get("@timestamp")
        if ts_str:
            dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            event_time = dt
        else:
            event_time = datum.event_time

        # 2. Extract Keys (belt_id) for partitioning
        belt_id = payload.get("belt_id", "GLOBAL")
        new_keys = [belt_id]

        return Messages(Message(keys=new_keys, value=datum.value, event_time=event_time))
    
    except Exception as e:
        print(f"Error in transformer: {e}")
        return Messages(Message(keys=keys, value=datum.value, event_time=datum.event_time))

if __name__ == "__main__":
    print("Starting Source Transformer (pynumaflow 0.12.1)...")
    try:
        SourceTransformServer(transform_handler).start()
    except grpc.RpcError as e:
        # The Rust numa sidecar closed the gRPC channel (e.g. ISB backpressure or
        # sidecar restart). Exit cleanly so Kubernetes restarts us without a crash.
        print(f"gRPC channel closed by numa sidecar (expected on sidecar restart): {e}")
        sys.exit(0)
