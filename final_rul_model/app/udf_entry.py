# app/udf_entry.py

import sys
import json
import traceback
import grpc
from pynumaflow.mapper import Messages, Message, Datum, MapServer
from app.runtime import MLRuntime

_runtime = None

def get_runtime():
    global _runtime
    if _runtime is None:
        _runtime = MLRuntime()
    return _runtime

def handler(keys: list, datum: Datum) -> Messages:
    try:
        if not datum.value:
            return Messages(Message.to_all(b"{}"))

        raw_event = json.loads(datum.value.decode("utf-8"))
        result = get_runtime().process(raw_event)

        return Messages(Message(value=json.dumps(result).encode("utf-8")))

    except Exception as e:
        error_payload = json.dumps({
            "status": "error",
            "message": str(e),
            "trace": traceback.format_exc()
        }).encode("utf-8")
        return Messages(Message(value=error_payload))

if __name__ == "__main__":
    print("Starting Final RUL Model UDF Server...")
    try:
        MapServer(handler).start()
    except grpc.RpcError as e:
        print(f"gRPC channel closed: {e}")
        sys.exit(0)
