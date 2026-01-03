from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from app import call_model_with_retry

import json
import hashlib
import time

app = FastAPI()

# ----------------------------
# Logging helper
# ----------------------------
def log_interaction(payload, response_obj):
    record = {
        "ts": time.time(),
        "payload_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "response": response_obj
    }

    with open("cobra_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ----------------------------
# Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    response = call_model_with_retry(
        prompt=payload["prompt"],
        expected_domain=payload["expected_domain"],
        expected_phase=payload["expected_phase"],
        symbol_universe=payload.get("symbol_universe"),
        strict_schema=True,
    )

    log_interaction(payload, response)

    return response



