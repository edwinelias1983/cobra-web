from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from enum import Enum
from app import Domain

from app import (
    call_model_with_retry_v7,
    CobraState,
    v7_state_domain_label,
    InteractionMode,
    v7_requires_domain0b,
    v7_domain0b_response,
    v7_record_domain0b_answer,
    v7_domain0_response,
)

import json
import hashlib
import time
import sqlite3
import uuid
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

    if isinstance(state.interaction_mode, str):
        state.interaction_mode = InteractionMode(state.interaction_mode)

    if isinstance(state.current_domain, str):
        state.current_domain = Domain(state.current_domain)

    return state

def save_session_state(session_id: str, state: CobraState):
    data = dict(state.__dict__)

    if isinstance(data.get("interaction_mode"), Enum):
        data["interaction_mode"] = data["interaction_mode"].value
    if isinstance(data.get("current_domain"), Enum):
        data["current_domain"] = data["current_domain"].value

    state_json = json.dumps(data, ensure_ascii=False)

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

    # =====================================================
    # PAYLOAD NORMALIZATION
    # =====================================================

    normalized = {}

    normalized["session_id"] = payload.get("session_id")

    normalized["interaction_mode"] = (
        payload.get("interaction_mode")
        or payload.get("interactionMode")
    )

    normalized["want_to_understand"] = (
        payload.get("want_to_understand")
        or payload.get("want")
    )

    likes = payload.get("likes")
    if isinstance(likes, str):
        likes = [likes]
    normalized["likes"] = likes

    if "auditory_response" in payload:
        normalized["auditory_response"] = payload["auditory_response"]

    if "prompt" in payload:
        normalized["prompt"] = payload["prompt"]

    if "micro_response" in payload:
        normalized["micro_response"] = payload["micro_response"]

    payload = normalized

    # V7 HARD RULE: client may never control domain or phase
    payload.pop("domain", None)
    payload.pop("phase", None)


    session_id = payload.get("session_id")  # <-- ADDED (so except can safely attach it if generated)

    try:
        # ---------------------------
        # Load session (SERVER-OWNED session_id if missing)
        # ---------------------------
        session_id = payload.get("session_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            payload["session_id"] = session_id
        print("SESSION_ID USED:", session_id)

        state = load_session_state(session_id)

        # =====================================================
        # DOMAIN 0: Commit interaction mode ONCE (SERVER-OWNED)
        # =====================================================
        
        if not state.interaction_mode and payload.get("interaction_mode"):
            try:
                state.interaction_mode = InteractionMode(payload["interaction_mode"])
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid interaction_mode: {payload['interaction_mode']}"
                )
            
            print("STATE INTERACTION MODE:", state.interaction_mode)

        # =====================================================
        # V7 HARD MICRO-CHECK GATE — NO ADVANCE WITHOUT PASS
        # =====================================================
        
        if getattr(state, "awaiting_micro_check", False) and not payload.get("micro_response"):
            response = state.last_microcheck_response

            response["state"] = {
                "domain0_complete": bool(getattr(state, "domain0_complete", False)),
                "domain0b_complete": bool(getattr(state, "domain0b_complete", False)),
            }

            save_session_state(session_id, state)
            return response

        # =====================================================
        # V7 HARD GUARD — Domain 0 / 0B are write-once
        # =====================================================
        if state.domain0_complete:
            payload.pop("interaction_mode", None)
            payload.pop("want_to_understand", None)
            payload.pop("likes", None)

        if not v7_requires_domain0b(state):
            payload.pop("auditory_response", None)
        
        # -------------------------------------------------
        # V7 DOMAIN 0B ANSWER RECORD (SERVER-OWNED, TIMING FIX)
        # -------------------------------------------------
        if payload.get("auditory_response"):
            v7_record_domain0b_answer(state, payload["auditory_response"])

        # -------------------------------------------------
        # V7 DOMAIN 0B ANSWER RECORDING (SERVER-OWNED)
        # -------------------------------------------------
        
        if payload.get("auditory_response"):
            v7_record_domain0b_answer(state, payload["auditory_response"])
        
        # ---------------------------
        # Build prompt
        # ---------------------------
   
        prompt = payload.get("prompt", "")

        # DOMAIN 0: inject required answers so V7 can evaluate them
        if not state.domain0_complete:
            prompt = (
                f"Interaction mode: {payload.get('interaction_mode')}\n"
                f"What do you want to understand today?: {payload.get('want_to_understand')}\n"
                f"What do you naturally understand or like?: {payload.get('likes')}\n\n"
                + prompt
            )

        if payload.get("micro_response"):
            prompt += f"\n\nUser micro-check response:\n{payload['micro_response']}"
            if hasattr(state, "awaiting_micro_check"):
                state.awaiting_micro_check = False

        # =====================================================
        # V7 HARD GUARD — prevent Domain 0 / 0B reseeding via prompt
        # =====================================================
        if state.domain0_complete:
            # Prompt may continue conversation, but not reseed symbols
            pass

        # ---------------------------
        # Call V7 engine
        # ---------------------------
        expected_domain = server_expected_domain(state)
        expected_phase = "PHASE_2" if getattr(state, "phase2_active", False) else "PHASE_1"
        symbol_universe = server_symbol_universe(payload, state) or []

        response = call_model_with_retry_v7(
            prompt=prompt,
            state=state,
            expected_domain=expected_domain,
            expected_phase=expected_phase,
            symbol_universe=symbol_universe,
            )
        log_interaction(payload, response)

        # -------------------------------------------------
        # V7 DOMAIN 0 / 0B STATE COMMIT (SERVER-OWNED)
        # -------------------------------------------------
        
        # DOMAIN 0 completes when required inputs exist
        if not getattr(state, "domain0_complete", False):
            if (
                payload.get("interaction_mode")
                and payload.get("want_to_understand")
                and payload.get("likes")
            ):
                state.domain0_complete = True

        # DOMAIN 0B completes when auditory response exists
        if not getattr(state, "domain0b_complete", False):
            if payload.get("auditory_response"):
                state.domain0b_complete = True

        if isinstance(response.get("domain"), str):
            try:
                state.current_domain = Domain(response["domain"])
            except ValueError:
                pass

        # -------------------------------------------------
        # V7 PHASE COMMIT (SERVER-OWNED)
        # -------------------------------------------------

        if response.get("phase") == "PHASE_2":
            state.phase2_active = True

        # =====================================================
        # V7 MICRO-CHECK ARMING (SERVER-OWNED)
        # =====================================================
        
        if response.get("intent") == "MICRO_CHECK":
            state.last_microcheck_response = response
            state.awaiting_micro_check = True
        
        # -------------------------------------------------
        # V7 UI CONTRACT: surface server-owned state
        # -------------------------------------------------
        response["state"] = {
            "domain0_complete": bool(getattr(state, "domain0_complete", False)),
            "domain0b_complete": bool(getattr(state, "domain0b_complete", False)),
        }


        save_session_state(session_id, state)

        return response

    except Exception as e:
        return {
            "domain": "SERVER_ERROR",
            "intent": "ERROR",
            "text": f"Internal Server Error: {str(e)}"
        }

