"""
setup_kibana.py
---------------
Creates Kibana index patterns for the Belt ML Runtime project.
Run this script after the stack is up and port-forwards are active:
  - Kibana: localhost:5601
  - Elasticsearch: localhost:9200
"""

import requests
import os
import sys
import time

KIBANA_URL = os.getenv("KIBANA_URL", "http://localhost:5601")
ES_URL     = os.getenv("ES_URL",     "http://localhost:9200")

# All index patterns expected in this project
INDEX_PATTERNS = [
    {"title": "belt-index*",        "timeFieldName": "@timestamp"},
    {"title": "belt-raw-data*",     "timeFieldName": "@timestamp"},
    {"title": "belt-runtime-state", "timeFieldName": "last_prediction_timestamp"},
]

HEADERS = {
    "kbn-xsrf": "true",
    "Content-Type": "application/json",
}


def wait_for_kibana(retries: int = 60, delay: int = 5) -> bool:
    """Poll Kibana /api/status until it returns 'available'."""
    url = f"{KIBANA_URL}/api/status"
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                level = r.json().get("status", {}).get("overall", {}).get("level", "")
                if level == "available":
                    print(f"[OK] Kibana is ready (attempt {attempt})")
                    return True
                print(f"[waiting] Kibana status: {level} ({attempt}/{retries})")
            else:
                print(f"[waiting] Kibana HTTP {r.status_code} ({attempt}/{retries})")
        except requests.exceptions.ConnectionError as e:
            print(f"[waiting] Cannot connect to Kibana: {e} ({attempt}/{retries})")
        except Exception as e:
            print(f"[waiting] Unexpected error: {e} ({attempt}/{retries})")
        time.sleep(delay)
    print("[ERROR] Kibana did not become ready in time.")
    return False


def get_existing_patterns() -> set:
    """Return a set of existing Kibana index pattern titles."""
    url = f"{KIBANA_URL}/api/saved_objects/_find?type=index-pattern&per_page=100"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        objects = r.json().get("saved_objects", [])
        return {obj["attributes"]["title"] for obj in objects}
    except Exception as e:
        print(f"[WARN] Could not fetch existing patterns: {e}")
        return set()


def create_index_pattern(pattern: dict, existing: set):
    """Create a Kibana index pattern if one with that title does not exist."""
    title = pattern["title"]
    if title in existing:
        print(f"[SKIP] Already exists: {title}")
        return

    url     = f"{KIBANA_URL}/api/index_patterns/index_pattern"
    payload = {
        "index_pattern": {
            "title":         title,
            "timeFieldName": pattern["timeFieldName"],
        }
    }
    try:
        r = requests.post(url, headers=HEADERS, json=payload, timeout=10)
        if r.status_code in (200, 201):
            print(f"[CREATED] {title}")
        elif r.status_code == 409:
            print(f"[SKIP] Conflict (already exists): {title}")
        else:
            print(f"[ERROR] Failed [{r.status_code}]: {title}")
            print(f"        Response: {r.text[:300]}")
    except Exception as e:
        print(f"[ERROR] Exception creating {title}: {e}")


def verify_es_indices():
    """Check Elasticsearch indices have data and are healthy."""
    print("\n--- Elasticsearch Index Verification ---")
    expected = [
        ("belt-index",         "ML predictions (RUL + health score)"),
        ("belt-raw-data",      "Raw IoT sensor telemetry"),
        ("belt-runtime-state", "Belt state / last-seen offsets"),
    ]
    all_ok = True
    for idx, description in expected:
        try:
            # Health check
            health_r = requests.get(f"{ES_URL}/_cat/indices/{idx}?h=health,status,docs.count", timeout=5)
            count_r  = requests.get(f"{ES_URL}/{idx}/_count", timeout=5)

            if count_r.status_code == 200:
                count  = count_r.json().get("count", 0)
                health = health_r.text.strip() if health_r.status_code == 200 else "unknown"
                status = "[OK]" if count > 0 else "[WARN] empty"
                print(f"  {status:12s} {idx:28s} {count:>8,} docs   {health}   # {description}")
                if count == 0:
                    all_ok = False
            else:
                print(f"  [MISSING] {idx} — HTTP {count_r.status_code}")
                all_ok = False
        except Exception as e:
            print(f"  [ERROR]   {idx} — {e}")
            all_ok = False

    return all_ok


def verify_kibana_patterns():
    """List all registered Kibana index patterns and their time fields."""
    print("\n--- Kibana Index Pattern Verification ---")
    url = f"{KIBANA_URL}/api/saved_objects/_find?type=index-pattern&per_page=100"
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.raise_for_status()
        objects = r.json().get("saved_objects", [])
        if not objects:
            print("  [WARN] No index patterns found in Kibana.")
            return False
        for obj in objects:
            attr  = obj["attributes"]
            title = attr.get("title", "?")
            tf    = attr.get("timeFieldName", "(no time field)")
            print(f"  [OK]  {title:30s}  time_field={tf}")
        return True
    except Exception as e:
        print(f"  [ERROR] Could not list Kibana patterns: {e}")
        return False


def setup_kibana():
    print(f"\nConnecting to Kibana at {KIBANA_URL}...")
    if not wait_for_kibana():
        sys.exit(1)

    # --- Elasticsearch ---
    es_ok = verify_es_indices()

    # --- Kibana index patterns ---
    print("\n--- Kibana Index Pattern Setup ---")
    existing = get_existing_patterns()
    for pattern in INDEX_PATTERNS:
        create_index_pattern(pattern, existing)

    verify_kibana_patterns()

    print("\n=== Summary ===")
    print(f"  Elasticsearch indices : {'OK' if es_ok else 'Some issues found'}")
    print(f"  Kibana URL            : {KIBANA_URL}")
    print(f"  Elasticsearch URL     : {ES_URL}")
    print("  Done.\n")


if __name__ == "__main__":
    setup_kibana()
