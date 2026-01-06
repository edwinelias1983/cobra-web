from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app import (
    call_model_with_retry,          # V6 (kept, not removed)
    call_model_with_retry_v7,       # V7 ADD
    CobraState                      # V7 ADD
)

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
        # V7 ADD: per-session state storage
        conn.execute("""
        CREATE TABLE IF NOT EXISTS cobra_sessions (
            session_id TEXT PRIMARY KEY,
            state_json TEXT,
            updated_ts REAL
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
# V7 SESSION STATE HELPERS (ADD ONLY)
# ============================================================

def load_session_state(session_id: str) -> CobraState:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "SELECT state_json FROM cobra_sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cur.fetchone()

    if not row:
        return CobraState()

    try:
        data = json.loads(row[0])
        return CobraState(**data)
    except Exception:
        return CobraState()

def save_session_state(session_id: str, state: CobraState):
    state_json = json.dumps(state.__dict__, ensure_ascii=False)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO cobra_sessions (session_id, state_json, updated_ts)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id)
            DO UPDATE SET
                state_json = excluded.state_json,
                updated_ts = excluded.updated_ts
            """,
            (session_id, state_json, time.time())
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
        # ----------------------------------------------------
        # V7 ADD: load per-session state
        # ----------------------------------------------------
        session_id = payload.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")

        state = load_session_state(session_id)

        # ----------------------------------------------------
        # Support micro-check continuation (UNCHANGED)
        # ----------------------------------------------------
        prompt = payload.get("prompt", "")

        if payload.get("micro_response"):
            prompt += f"\n\nUser micro-check response:\n{payload['micro_response']}"

        # ----------------------------------------------------
        # V7 ADD: call V7 orchestrator (no removal of V6)
        # ----------------------------------------------------
        response = call_model_with_retry_v7(
            prompt=prompt,
            expected_domain=payload["expected_domain"],
            expected_phase=payload["expected_phase"],
            state=state,
            symbol_universe=payload.get("symbol_universe"),
        )

        # ----------------------------------------------------
        # V7 ADD: inject advance gate + persist state
        # ----------------------------------------------------
        if isinstance(response, dict):
            response.setdefault(
                "advance_allowed",
                bool(state.last_microcheck_passed and not state.consolidation_active)
            )

        save_session_state(session_id, state)

    except Exception as e:
        error_response = {
            "error": "backend_failure",
            "message": str(e)
        }
        log_interaction(payload, error_response)
        return error_response

    log_interaction(payload, response)
    return response
