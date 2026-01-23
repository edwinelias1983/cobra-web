from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
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
    v7_get_symbol_universe,
)
import os
from openai import OpenAI

import json
import hashlib
import time
import sqlite3
import uuid
from pathlib import Path

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        (session_id, state_json, time.time()),
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

def add_domain1_image_row_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 1 image generator: symbolic images from the user's symbols.
    """
    if response.get("domain") != "D1":
        return response

    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    # Avoid adding multiple image rows
    if any(isinstance(b, dict) and b.get("type") == "image_row" for b in blocks):
        return response

    # Prefer symbols_used for this turn; fall back to full universe
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    prompt_text = (
        "Create a simple, clean symbolic image using ONLY these elements from the learner's world: "
        + ", ".join(focus)
        + ". No text, no extra characters, no new franchises."
    )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt_text,
            n=1,
            size="1024x1024",
        )
        url = img_resp.data[0].url
    except Exception:
        # If the image API fails, keep the response unchanged
        return response

    blocks.insert(
        0,
        {
            "type": "image_row",
            "images": [
                {
                    "src": url,
                    "alt": f"Symbolic visual for: {', '.join(focus)}",
                }
            ],
        },
    )

    payload["blocks"] = blocks
    response["payload"] = payload
    return response

def add_domain2_images_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 2 image generator: metaphoric scenes from the user's symbols.
    """
    if response.get("domain") != "D2":
        return response

    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    # Avoid stacking multiple image rows
    if any(isinstance(b, dict) and b.get("type") == "image_row" for b in blocks):
        return response

    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    prompt_text = (
        "Create a metaphoric scene that uses ONLY these elements from the learner's world: "
        + ", ".join(focus)
        + ". The scene should suggest relationships and dynamics, but must not add new shows, teams, or symbols."
    )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt_text,
            n=1,
            size="1024x1024",
        )
        url = img_resp.data[0].url
    except Exception:
        return response

    blocks.insert(
        0,
        {
            "type": "image_row",
            "images": [
                {
                    "src": url,
                    "alt": f"Metaphoric scene using: {', '.join(focus)}",
                }
            ],
        },
    )

    payload["blocks"] = blocks
    response["payload"] = payload
    return response

def add_domain3_diagram_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 3 diagram generator: simple pattern diagrams, no photos/GIFs.
    """
    if response.get("domain") != "D3":
        return response

    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    # Avoid multiple diagrams
    if any(isinstance(b, dict) and b.get("type") == "diagram" for b in blocks):
        return response

    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    prompt_text = (
        "Create a clean, minimal diagram (boxes and arrows only, no characters, no photos, no GIFs) "
        "that shows patterns and relationships between these elements from the learner's world: "
        + ", ".join(focus)
        + ". White background, simple lines, high contrast."
    )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt_text,
            n=1,
            size="1024x1024",
        )
        url = img_resp.data[0].url
    except Exception:
        return response

    blocks.insert(
        0,
        {
            "type": "diagram",
            "src": url,
            "alt": f"Pattern diagram for: {', '.join(focus)}",
        },
    )

    payload["blocks"] = blocks
    response["payload"] = payload
    return response

def ensure_domain1_structure(response: dict) -> dict:
    """
    Make sure Domain 1 responses have the structured payload and micro_check fields
    that the V7 UI expects, and inject the Domain 1 mapping text.
    """
    if response.get("domain") != "D1":
        return response

    # 1) Ensure DOMAIN 1 symbolic visual block (what you already had)
    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    if not blocks:
        # Keep your existing text structure
        blocks = [
            {
                "type": "header",
                "text": "DOMAIN 1 — SYMBOLIC (PRIMARY VISUAL LAYER)",
            },
            {
                "type": "subtext",
                "text": (
                    "I’ll introduce only simple symbols, pulled directly from your world. "
                    "No explanations yet — just anchors."
                ),
            },
            # TEMP: keep your current images for now
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

    # 2) NEW: append “Quantum physics — stripped…” mapping into response["text"]
    text = (response.get("text") or "").strip()
    mapping_header = "Quantum physics — stripped of jargon:"

    # Avoid duplicating the mapping if the model ever echoes it
    if mapping_header not in text:
        mapping_lines = [
            "",
            mapping_header,
            "",
            "- System = the whole Sopranos family network or the entire Barça squad on the pitch.",
            "- State = how things are right now — who is where, who is stable or under pressure.",
            "- Interaction = the concrete moves: a conversation in therapy, a pass between players, a sudden shift in trust.",
            "- Constraint = the rules, debts, loyalties, and tactics that limit what anyone can do next.",
            "- Uncertainty = what nobody can fully predict — who will flip, where the ball will actually end up, which plan breaks.",
            "- Observation = the moment something becomes visible: the therapist hearing the truth, the camera seeing a play, the crowd finally understanding the pattern.",
            "",
            "We stay inside these symbols only; no physics language yet.",
        ]
        mapping_block = "\n".join(mapping_lines)

        if text:
            text = text + "\n\n" + mapping_block
        else:
            text = mapping_block

        response["text"] = text

    # 3) Ensure Domain 1 micro-check scaffold (unchanged)
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
        'In your words: what is a "state" — without using science terms?',
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
def add_domain1_image_row_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Inject a Domain 1 image_row based on the user's symbolic universe.
    TEMP: placeholder that will later connect to a real image generator.
    """
    if response.get("domain") != "D1":
        return response

    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    # If an image_row already exists, do nothing for now
    if any(isinstance(b, dict) and b.get("type") == "image_row" for b in blocks):
        return response

    su = getattr(state, "symbolic_universe", None)
    label = "your symbols"
    if isinstance(su, dict):
        symbols = su.get("symbols") or su.get("symbol_universe") or []
        label = ", ".join(str(x) for x in symbols) or label
    elif isinstance(su, list):
        label = ", ".join(str(x) for x in su) or label

    blocks.insert(
        0,
        {
            "type": "image_row",
            "images": [
                {
                    "src": "symbol-universe-placeholder.jpg",
                    "alt": f"Visual anchored in: {label}",
                }
            ],
        },
    )

    payload["blocks"] = blocks
    response["payload"] = payload
    return response

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

        ## ---------------------------
        # Call V7 engine
        # ---------------------------
        expected_domain = server_expected_domain(state)

        # HARD OVERRIDE: once Domain 0 is complete, push into Domain 1
        if getattr(state, "domain0_complete", False):
            expected_domain = Domain.D1

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

        # If server expects D1 but model still said D0, coerce to D1
        if getattr(state, "domain0_complete", False) and response.get("domain") == "D0":
            response["domain"] = "D1"

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
        # SERVER FALLBACK: First D1 micro-check if model forgets
        # =====================================================
        if (
            getattr(state, "domain0_complete", False)
            and response.get("domain") == "D1"
            and not getattr(state, "domain1_microcheck_shown", False)
            and response.get("intent") != "MICRO_CHECK"
        ):
            # Let ensure_domain1_structure create / normalize micro_check
            # Customize Domain 1 visuals from the user's symbolic universe
            response = add_domain1_image_row_from_symbols(response, state)

            # Domain 1: inject images from the user's symbol universe
            response = add_domain1_image_row_from_symbols(response, state)

            # Ensure Domain 1 blocks + micro_check shape
            response = ensure_domain1_structure(response)

            # Domain 1: inject images from the user's symbol universe
            response = add_domain1_image_row_from_symbols(response, state)

            # Ensure Domain 1 blocks + micro_check shape
            response = ensure_domain1_structure(response)

            # Domain 2: inject metaphoric images from the user's symbol universe
            response = add_domain2_images_from_symbols(response, state)

            # Domain 3: inject a simple diagram based on the user's symbols
            response = add_domain3_diagram_from_symbols(response, state

            mc = response.get("micro_check") or {}
            mc.setdefault("required", True)
            mc.setdefault("rules", [
                "One sentence.",
                "Use only Sopranos or Barça language.",
                "No science terms yet.",
            ])
            mc.setdefault(
                "prompt",
                'In your words: what is a "state" — without using science terms?',
            )
            response["micro_check"] = mc

            response["intent"] = "MICRO_CHECK"
            state.domain1_microcheck_shown = True
            state.last_microcheck_response = response
            state.awaiting_micro_check = True

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

        # -------------------------------------------------
        # NEW: Domain 1 intro preface on first D1 turn
        # -------------------------------------------------
        if (
            response.get("domain") == "D1"
            and not getattr(state, "phase2_active", False)  # still in PHASE_1
            and not getattr(state, "domain1_intro_shown", False)
        ):
            # Build the intro from state
            symbolic_universe_label = " + ".join(
                getattr(state, "symbol_universe_labels", [])
            ) if getattr(state, "symbol_universe_labels", None) else "your symbols"

            mode = getattr(state, "interaction_mode", None)
            mode_label = (
                mode.value.capitalize()
                if hasattr(mode, "value")
                else str(mode) if mode else "Unknown"
            )

            intro_lines = [
                "Good. Domain 0 is now locked.",
                f"Symbolic universe = {symbolic_universe_label}",
                f"Mode = {mode_label}",
                "",
                "We move bottom-up.",
                "",
            ]

            intro_text = "\n".join(intro_lines)
            existing_text = response.get("text") or ""
            response["text"] = intro_text + existing_text

            # Remember that we already showed the intro once
            state.domain1_intro_shown = True

        # Ensure Domain 1 blocks + micro_check shape
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

