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
    """
    Server-owned symbolic universe:
    - Before Domain 0 is complete: accept from payload and commit into state.
    - After Domain 0 is complete: ignore payload and use stored state only.
    """
    # Before Domain 0 is complete, allow payload to set it
    if not getattr(state, "domain0_complete", False) and "symbol_universe" in payload:
        return payload["symbol_universe"]

    # After Domain 0 is complete, ignore payload and use stored state
    if isinstance(state.symbolic_universe, dict):
        return (
            state.symbolic_universe.get("symbol_universe")
            or state.symbolic_universe.get("symbols")
        )

    if isinstance(state.symbolic_universe, list):
        return state.symbolic_universe

    return None


def ensure_domain1_structure(response: dict) -> dict:
    """
    Make sure Domain 1 responses have the structured payload and micro_check fields
    that the V7 UI expects.
    """
    if response.get("domain") != "D1":
        return response

    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    if not blocks:
        blocks = [
            {
                "type": "header",
                "text": "DOMAIN 1 — SYMBOLIC (PRIMARY VISUAL LAYER)"
            },
            {
                "type": "subtext",
                "text": (
                    "I’ll introduce only simple symbols, pulled directly from your world. "
                    "No explanations yet — just anchors."
                )
            },
            {
                "type": "image_row",
                "images": [
                    {"src": "sopranos-therapy.jpg", "alt": "Sopranos therapy room"},
                    {"src": "family-chart.png", "alt": "Family structure chart"},
                    {"src": "barca-passing.png", "alt": "Barça passing map"},
                ],
            },
            {
                "type": "grouped_list",
                "title": "From The Sopranos",
                "items": [
                    "The Family → a closed system",
                    "Tony → central node under pressure",
                    "Crews → semi-independent units",
                    "Therapy room → hidden/internal layer",
                    "Trust / betrayal → unstable alignment",
                ],
            },
            {
                "type": "grouped_list",
                "title": "From FC Barcelona",
                "items": [
                    "The Ball → information / state",
                    "Passing → interaction",
                    "Midfield triangle → coordination under constraints",
                    "Possession → control without force",
                    "System > individual → structure dominates outcomes",
                ],
            },
            {
                "type": "instruction",
                "text": (
                    "Using only your symbols, notice: system, state, interaction, "
                    "constraint, uncertainty, observation."
                ),
            },
        ]

    payload["blocks"] = blocks
    response["payload"] = payload

    micro_check = response.get("micro_check") or {}
    micro_check.setdefault("required", True)
    micro_check.setdefault(
        "rules",
        [
            "One sentence.",
            "Use only Sopranos or Barça language.",
            "No science terms yet.",
        ],
    )
    micro_check.setdefault(
        "prompt",
        'In your words: what is a "state" — without using science terms?'
    )
    response["micro_check"] = micro_check

    return response

def domain1_style_instruction() -> str:
    """
    Instruction text enforcing 'Domain 1 = symbols only, no theory'.
    """
    return (
        "You are in DOMAIN 1 — SYMBOLIC LAYER.\n"
        "RULES:\n"
        "- Use ONLY simple symbols and tokens from the user's symbolic universe.\n"
        "- NO physics terminology.\n"
        "- NO formal theory, equations, or explanations.\n"
        "- Describe Sopranos and Barça tokens only: family, Tony, crews, therapy room, "
        "trust/betrayal, ball, passing, midfield triangle, possession, system > individual.\n"
        "- Do not explain quantum physics yet; only map these tokens.\n"
    )

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
        # If awaiting_micro_check is set (e.g., from Domain 1), always return the same
        # micro-check payload until a micro_response is provided.

        if getattr(state, "awaiting_micro_check", False) and not payload.get("micro_response"):
            response = state.last_microcheck_response
            response.setdefault("intent", "MICRO_CHECK")
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

        # A2: Enforce 'Domain 1 = NO THEORY'
        if expected_domain == Domain.D1:
            prompt = domain1_style_instruction() + "\n\n" + prompt

        response = call_model_with_retry_v7(
            prompt=prompt,
            state=state,
            expected_domain=expected_domain,
            expected_phase=expected_phase,
            symbol_universe=symbol_universe,
            )
        log_interaction(payload, response)

            # A4: Commit symbolic universe to state when Domain 0 completes
        if not getattr(state, "domain0_complete", False) and response.get("domain") == "D0":
            su = payload.get("symbol_universe")
            if su is not None:
                state.symbolic_universe = su
        # Assume response may mark domain0_complete; honor that
            if response.get("state", {}).get("domain0_complete"):
                state.domain0_complete = True

        # -------------------------------------------------
        # V7 DOMAIN 0 / 0B STATE COMMIT (SERVER-OWNED)
        # -------------------------------------------------

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

        response = ensure_domain1_structure(response)

        # C2: ensure response has stable shape
        response.setdefault("payload", {})
        response.setdefault("micro_check", {})
        response.setdefault("text", "")
        response.setdefault("domain", server_expected_domain(state))
        response.setdefault("intent", response.get("intent", "NORMAL"))

        save_session_state(session_id, state)

        return response

    except Exception as e:
        return {
            "domain": "SERVER_ERROR",
            "intent": "ERROR",
            "text": f"Internal Server Error: {str(e)}"
        }

