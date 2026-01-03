from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app import call_model_with_retry

import json
import hashlib
import time
import sqlite3
from pathlib import Path

app = FastAPI()

# ============================================================
# PERSISTENT STORAGE (SQLite)
# ============================================================

DB_PATH = Path("cobra_data.sqlite")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL,
            payload_hash TEXT,
            payload TEXT,
            response TEXT
        )
        """)
        conn.commit()

init_db()

def log_interaction(payload, response_obj):
    payload_json = json.dumps(payload, ensure_ascii=False)
    response_json = json.dumps(response_obj, ensure_ascii=False)

    payload_hash = hashlib.sha256(
        payload_json.encode("utf-8")
    ).hexdigest()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO interactions (ts, payload_hash, payload, response)
            VALUES (?, ?, ?, ?)
            """,
            (time.time(), payload_hash, payload_json, response_json)
        )
        conn.commit()

# ============================================================
# Routes
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    try:
        response = call_model_with_retry(
            prompt=payload["prompt"],
            expected_domain=payload["expected_domain"],
            expected_phase=payload["expected_phase"],
            symbol_universe=payload.get("symbol_universe"),
            strict_schema=True,
        )

    except Exception as e:
        # IMPORTANT: always return JSON
        error_response = {
            "error": "backend_failure",
            "message": str(e)
        }
        log_interaction(payload, error_response)
        return error_response

    log_interaction(payload, response)
    return response






