from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app import (
    call_model_with_retry,          # V6 (kept, not removed)
    call_model_with_retry_v7,       # V7
    CobraState,                     # V7
    v7_state_domain_label           # V7: canonical domain label from server state
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
# V7 SESSION STATE HELPERS
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
        state = CobraState()
        for k, v in data.items():
            if hasattr(state, k):
                setattr(state, k, v)
        return state
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
# V7 SERVER-AUTHORITATIVE DOMAIN/PHASE
# ============================================================

def server_expected_domain(state: CobraState) -> str:
    # Canonical V7 label from server state (D0/D0B/D1/...)
    return v7_state_domain_label(state)

def server_expected_phase(state: CobraState) -> str:
    # Server owns phase. Phase 2 only after Phase 1 transfer complete.
    if getattr(state, "phase1_transfer_complete", False):
        return "PHASE_2"
    return "PHASE_1"

def server_symbol_universe(payload: dict, state: CobraState):
    # Prefer payload if provided, else attempt to derive from state (if stored there).
    su = payload.get("symbol_universe")
    if su is not None:
        return su
    if isinstance(state.symbolic_universe, dict):
        # common storage patterns; falls back safely to None
        return (
            state.symbolic_universe.get("symbol_universe")
            or state.symbolic_universe.get("symbols")
            or None
        )
    if isinstance(state.symbolic_universe, list):
        return state.symbolic_universe
    return None

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
        # V7: load per-session state (required)
        # ----------------------------------------------------
        session_id = payload.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")

        state = load_session_state(session_id)

        # ----------------------------------------------------
        # Build prompt (micro-check continuation supported)
        # ----------------------------------------------------
        prompt = payload.get("prompt", "")
        if payload.get("micro_response"):
            prompt += f"\n\nUser micro-check response:\n{payload['micro_response']}"

        # ----------------------------------------------------
        # V7: SERVER-AUTHORITATIVE expected_domain + expected_phase
        # Ignore any client-supplied expected_domain/expected_phase
        # ----------------------------------------------------
        expected_domain = server_expected_domain(state)
        expected_phase = server_expected_phase(state)

        # ----------------------------------------------------
        # V7: call orchestrator
        # ----------------------------------------------------
        response = call_model_with_retry_v7(
            prompt=prompt,
            expected_domain=expected_domain,
            expected_phase=expected_phase,
            state=state,
            symbol_universe=server_symbol_universe(payload, state),
        )

        # ----------------------------------------------------
        # V7: advance gate hardening (locking intents)
        # ----------------------------------------------------
        if isinstance(response, dict):
            intent = response.get("intent")
            locking_intents = {
                "MICRO_CHECK",
                "REPAIR",
                "TRANSFER_CHECK",
                "STRESS_TEST",
                "PHASE2_CHOICE",
            }

            if intent in locking_intents:
                response["advance_allowed"] = False
            else:
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
