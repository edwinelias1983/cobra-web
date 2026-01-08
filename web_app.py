from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app import (
    call_model_with_retry_v7,   # V7 orchestrator
    CobraState,
    v7_state_domain_label,
    InteractionMode,
    v7_requires_domain0b,
    v7_domain0b_response,
    v7_record_domain0b_answer,
    v7_domain0_response,        # MISSING IMPORT (FIXED)
)

import json
import hashlib
import time
import sqlite3
from pathlib import Path

app = FastAPI()
print("WEB_APP_REACHED_FASTAPI")
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

    payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

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
# SESSION STATE HELPERS
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

    data = json.loads(row[0])
    state = CobraState()
    for k, v in data.items():
        if hasattr(state, k):
            setattr(state, k, v)
    return state

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
# SERVER-AUTHORITATIVE DOMAIN / PHASE
# ============================================================

def server_expected_domain(state: CobraState) -> str:
    return v7_state_domain_label(state)

def server_expected_phase(state: CobraState) -> str:
    return "PHASE_2" if getattr(state, "phase1_transfer_complete", False) else "PHASE_1"

def server_symbol_universe(payload: dict, state: CobraState):
    if "symbol_universe" in payload:
        return payload["symbol_universe"]
    if isinstance(state.symbolic_universe, dict):
        return (
            state.symbolic_universe.get("symbol_universe")
            or state.symbolic_universe.get("symbols")
        )
    if isinstance(state.symbolic_universe, list):
        return state.symbolic_universe
    return None

# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/cobra/run")
def run_cobra(payload: dict):
    try:
        # ---------------------------
        # Load session
        # ---------------------------
        session_id = payload.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="Missing session_id")

        state = load_session_state(session_id)

        # =====================================================
        # V7 HARD GUARD — Domain 0 / 0B are write-once
        # =====================================================
        if state.domain0_complete:
            payload.pop("interaction_mode", None)
            payload.pop("want_to_understand", None)
            payload.pop("likes", None)

        if not v7_requires_domain0b(state):
            payload.pop("auditory_response", None)

        # =====================================================
        # V7 DOMAIN 0 — SERVER-OWNED, ONE-TIME INITIALIZATION
        # =====================================================
        if not state.domain0_complete:
            if not all(k in payload for k in ("interaction_mode", "want_to_understand", "likes")):
                response = v7_domain0_response()
                log_interaction(payload, response)
                return response

            state.interaction_mode = InteractionMode(payload["interaction_mode"])
            state.symbolic_universe["domain0"] = {
                "want_to_understand": payload["want_to_understand"],
                "likes": payload["likes"],
            }
            state.domain0_complete = True
            save_session_state(session_id, state)

        # =====================================================
        # V7 DOMAIN 0B — AUDITORY SYMBOL MAP (ENFORCED)
        # =====================================================
        if v7_requires_domain0b(state):
            if "auditory_response" in payload:
                v7_record_domain0b_answer(state, payload["auditory_response"])
                save_session_state(session_id, state)

            response = v7_domain0b_response(state)
            if response:
                log_interaction(payload, response)
                return response
        # else: Domain 0B complete → fall through

        # ---------------------------
        # Build prompt
        # ---------------------------
        prompt = payload.get("prompt", "")
        if payload.get("micro_response"):
            prompt += f"\n\nUser micro-check response:\n{payload['micro_response']}"

        # ---------------------------
        # Server-authoritative control
        # ---------------------------
        expected_domain = server_expected_domain(state)
        expected_phase = server_expected_phase(state)

        # ---------------------------
        # Call V7 engine
        # ---------------------------
        response = call_model_with_retry_v7(
            prompt=prompt,
            state=state,
            expected_domain=expected_domain,
            expected_phase=expected_phase,
            symbol_universe=server_symbol_universe(payload, state),
        )

        # =====================================================
        # V7 HARD LOCK — server enforces non-advancement
        # =====================================================
        if isinstance(response, dict):
            if response.get("intent") in {
                "MICRO_CHECK",
                "REPAIR",
                "TRANSFER_CHECK",
                "STRESS_TEST",
                "PHASE2_CHOICE",
            }:
                response["advance_allowed"] = False

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
