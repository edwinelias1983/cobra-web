from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from enum import Enum
from app import Domain

from app import (
    Domain,
    call_model_with_retry_v7,
    CobraState,
    v7_state_domain_label,
    InteractionMode,
    v7_requires_domain0b,
    v7_domain0b_response,
    v7_record_domain0b_answer,
    v7_domain0_response,
    v7_get_symbol_universe,
    v7_phase2_stress_test_required,
    v7_phase2_stress_test_prompt,
    v7_phase1_transfer_required,
    v7_phase1_transfer_response,
    v7_expansion_prompt,  
)

import os
from openai import OpenAI

import json
import hashlib
import time
import sqlite3
import uuid
import copy
from pathlib import Path

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return {"status": "ok"}

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

    session_id = payload.get("session_id") or str(uuid.uuid4())

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO interactions (ts, payload_hash, payload, response)
            VALUES (?, ?, ?, ?)
            """,
            (time.time(), payload_hash, payload_json, response_json),
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
    """
    Server-owned domain scheduler, mirroring V7:

    Phase 1 (bottom-up, PHASE_1):
        D1 → D2 → D2B → D3 → D3B → D4 → D5

    Phase 2 (top-down, PHASE_2):
        D4 → D3B → D3 → D2B → D2 → D1
    """
    # If Domain 0/0B not complete yet, defer to V7's own label
    if not getattr(state, "domain0_complete", False):
        return v7_state_domain_label(state)

    # Canonical sequences (must match app.py)
    phase1_seq = ["D1", "D2", "D2B", "D3", "D3B", "D4", "D5"]
    phase2_seq = ["D4", "D3B", "D3", "D2B", "D2", "D1"]

    # Normalize current domain to simple string
    current = getattr(state, "current_domain", "D1")
    if isinstance(current, Domain):
        current = current.value

    # PHASE 2: top-down inversion once phase2_active is set
    if getattr(state, "phase2_active", False):
        # If somehow outside sequence, start at top (D4)
        if current not in phase2_seq:
            return Domain.D4
        idx = phase2_seq.index(current)
        # Stay at D1 when sequence is exhausted
        if idx + 1 >= len(phase2_seq):
            return Domain.D1
        return Domain(phase2_seq[idx + 1])

    # PHASE 1: bottom-up construction (default)
    # If somehow outside sequence, start at D1
    if current not in phase1_seq:
        return Domain.D1
    idx = phase1_seq.index(current)
    # Stay at D5 when sequence is exhausted
    if idx + 1 >= len(phase1_seq):
        return Domain.D5
    return Domain(phase1_seq[idx + 1])

def normalize_symbol_universe(payload: dict) -> list:
    """
    Server-authoritative symbol extraction.
    Domain 0 only. No side effects.
    """
    symbols = []

    su = payload.get("symbol_universe")
    if isinstance(su, list):
        symbols.extend(su)

    likes = payload.get("likes")
    if isinstance(likes, str):
        likes = [likes]
    if isinstance(likes, list):
        symbols.extend(likes)

    cleaned = []
    for s in symbols:
        if isinstance(s, str):
            s = s.strip()
            if s and s not in cleaned:
                cleaned.append(s)

    return cleaned

def server_symbol_universe(payload: dict, state: CobraState):
    """
    V7 HARD GUARANTEE:
    - Domain 0 completion => non-empty symbol universe
    - After lock, payload is ignored
    """

    reset_symbols = bool(payload.get("reset_symbols"))

    # -------------------------
    # DOMAIN 0 (UNLOCKED)
    # -------------------------
    if not getattr(state, "domain0_complete", False) or reset_symbols:

        symbols = normalize_symbol_universe(payload)

        if symbols:
            state.symbolic_universe["symbol_universe"] = symbols
            return symbols

        # Domain 0 attempted but invalid → HARD REPAIR
        return None

    # -------------------------
    # DOMAIN 0 LOCKED
    # -------------------------
    su = state.symbolic_universe.get("symbol_universe")
    if isinstance(su, list) and su:
        return su
        
    # IMPOSSIBLE STATE → FORCE REPAIR
    state.domain0_complete = False
    return None


def enforce_symbol_scope(response: dict, state: CobraState) -> dict:
    """
    Enforce that symbols_used is a subset of the user's symbolic universe.
    If missing, default to the full universe.
    """
    symbols = v7_get_symbol_universe(state) or []
    if not symbols:
        return response

    used = response.get("symbols_used")

    # If model didn't declare symbols_used, force it
    if not used:
        response["symbols_used"] = symbols
        return response

    # Filter to allowed symbols only
    filtered = [s for s in used if s in symbols]

    # If model tried to introduce new symbols, snap back
    response["symbols_used"] = filtered if filtered else symbols
    return response

def remove_placeholder_images(blocks: list) -> list:
    """
    V7 HARD RULE:
    No placeholder or legacy images may survive after Domain 0.
    """
    cleaned = []

    for b in blocks:
        if (
            isinstance(b, dict)
            and b.get("type") == "image_row"
            and any(
                img.get("src") == "symbol-universe-placeholder.jpg"
                for img in b.get("images", [])
            )
        ):
            continue  # DROP legacy placeholder image rows

        cleaned.append(b)

    return cleaned

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
        "Create a literal symbolic image with ONLY the following items:\n"
        + ", ".join(focus)
        + "\n\nRULES:\n"
        "- Each symbol appears exactly once.\n"
        "- No background scenery.\n"
        "- No implied story or action.\n"
        "- No extra objects, people, or context.\n"
        "- Flat, neutral background.\n"
        "- Diagrammatic, not artistic.\n"
        "- Do not add meaning beyond the listed symbols."
    )

    try:
        img_resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt_text,
            n=1,
            size="1024x1024",
        )
        url = img_resp.data[0].url
    except Exception as e:
        print("Image generation failed:", e)
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

def add_domain1_explanation_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 1 explanation: text built from the user's symbols for this turn.
    """
    if response.get("domain") != "D1":
        return response

    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    # Prefer symbols_used this turn, fall back to full universe
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    summary = (
        "In your symbolic universe, this step is about turning things you already know and care about "
        f"({', '.join(focus)}) into clean visual tokens that the system can move around."
    )

    mapping = {
        "system": (
            "The system is the whole setup that can notice, store, and reuse these symbols from your world."
        ),
        "state": (
            "The state is the current configuration of those symbols—what is active, combined, or being focused on right now."
        ),
        "interaction": (
            "The interaction is how your answers change which symbols are highlighted or connected, one step at a time."
        ),
    }

    # Store structured explanation (internal, non-UI)
    response["symbolic_explanation"] = {
        "summary": summary,
        "focus_symbols": focus,
        "mapping": mapping,
    }

    # ALSO surface explanation visibly (V7 UI contract)
    payload = response.get("payload") or {}
    blocks = payload.get("blocks") or []

    # Avoid duplicate explanation blocks
    if not any(
        isinstance(b, dict) and b.get("type") == "subtext" and "Symbolic explanation:" in b.get("text", "")
        for b in blocks
    ):
        blocks.append(
            {
                "type": "subtext",
                "text": "Symbolic explanation:\n" + summary,
            }
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
def add_domain2b_explanation_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 2B — analogical structure mapping explanation,
    built entirely from the learner's symbolic universe.
    """
    # Only run in Domain 2B
    if response.get("domain") != "D2B":
        return response

    # Get the user's symbol universe from state
    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    # Prefer symbols_used this turn; otherwise use all symbols
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    # Core explanation text for Domain 2B
    summary = (
        "Here we take one situation from your world and line it up with a new one, "
        "so that the roles, pressures, and possible moves match across both stories. "
        f"We stay inside your symbols ({', '.join(focus)}) while we do that mapping."
    )

    # Attach a clean, predictable field into the response
    response["domain2b_explanation"] = {
        "summary": summary,
        "focus_symbols": focus,
        "kind": "analogical_structure_mapping",
    }

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

def add_domain3b_explanation_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 3B — temporal / narrative explanation,
    using only the learner's symbolic universe.
    """
    # Only run in Domain 3B
    if response.get("domain") != "D3B":
        return response

    # Get the user's symbol universe from state
    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    # Prefer symbols_used this turn; otherwise use all symbols
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    summary = (
        "Here we tell a before-and-after story entirely inside your world, "
        "showing how things start, what pushes them to change, and what they can turn into next. "
        f"We track those shifts using your symbols ({', '.join(focus)})."
    )

    response["domain3b_explanation"] = {
        "summary": summary,
        "focus_symbols": focus,
        "kind": "temporal_narrative",
    }

    return response

def add_domain4_explanation_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 4 — systemic / constraint-based explanation,
    using only the learner's symbolic universe.
    """
    # Only run in Domain 4
    if response.get("domain") != "D4":
        return response

    # Get the user's symbol universe from state
    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    # Prefer symbols_used this turn; otherwise use all symbols
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    summary = (
        "Here we step back and talk about the whole setup in your world: "
        "what rules, limits, and background conditions shape what can actually happen. "
        f"We describe those constraints using your symbols ({', '.join(focus)})."
    )

    response["domain4_explanation"] = {
        "summary": summary,
        "focus_symbols": focus,
        "kind": "systemic_constraints",
    }

    return response

def add_domain5_explanation_from_symbols(response: dict, state: CobraState) -> dict:
    """
    Domain 5 — paradox / tension framing,
    using only the learner's symbolic universe.
    """
    # Only run in Domain 5
    if response.get("domain") != "D5":
        return response

    # Get the user's symbol universe from state
    symbols = v7_get_symbol_universe(state)
    if not symbols:
        return response

    # Prefer symbols_used this turn; otherwise use all symbols
    used = response.get("symbols_used") or symbols
    focus = [str(s) for s in used] or [str(s) for s in symbols]

    summary = (
        "Here we name the tensions inside your world that never fully go away—"
        "situations where two things stay true at the same time and keep pulling on each other. "
        f"We describe those live contradictions using your symbols ({', '.join(focus)})."
    )

    response["domain5_explanation"] = {
        "summary": summary,
        "focus_symbols": focus,
        "kind": "paradox_tension",
    }

    return response

def ensure_domain1_structure(response: dict, state: CobraState) -> dict:

    # CHANGE 3: Do not mutate Domain-1 once symbolic blocks are locked
    if response.get("_symbolic_blocks_locked"):
        return response

    """
    Make sure Domain 1 responses have the structured payload and micro_check fields
    that the V7 UI expects, and inject the Domain 1 mapping text.
    """
    if response.get("domain") != "D1":
        return response

    # 1) Ensure DOMAIN 1 symbolic visual block
    payload = response.get("payload") or {}
    blocks = payload.get("blocks")

    if not blocks:
        payload["blocks"] = [
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
            {
                "type": "grouped_list",
                "title": "Your symbols",
                "items": [
                    "We’ll keep using only the symbols you named in Domain 0.",
                    "Every visual and explanation will stay inside that symbolic universe.",
                ],
            },
            {
                "type": "instruction",
                "text": (
                    "Using only your symbols, notice how things are arranged, "
                    "what can change, and what limits those changes."
                ),
            },
        ]

        response["payload"] = payload

    # 2) Append symbolic mapping into response["text"]
    text = (response.get("text") or "").strip()
    mapping_header = "Quantum physics — stripped of jargon:"

    if mapping_header not in text:
        symbols = v7_get_symbol_universe(state) or ["your world"]
        symbol_label = ", ".join(symbols)

        mapping_lines = [
            "",
            "Symbolic mapping (using only your world):",
            "",
            f"- System = the whole setup inside {symbol_label}.",
            "- State = how things are arranged right now.",
            "- Interaction = the moves that change the situation.",
            "- Constraint = what limits or shapes those moves.",
            "- Uncertainty = what can’t be fully predicted yet.",
            "- Observation = the moment something becomes clear.",
            "",
            "We stay inside these symbols only.",
        ]

        mapping_block = "\n".join(mapping_lines)
        response["text"] = text + "\n\n" + mapping_block if text else mapping_block
    else:
        symbols = v7_get_symbol_universe(state) or ["your world"]
        symbol_label = ", ".join(symbols)

    # 3) Ensure Domain 1 micro-check scaffold
    micro_check = response.get("micro_check") or {}
    micro_check.setdefault("required", True)
    micro_check.setdefault(
        "rules",
        [
            "One sentence.",
            f"Use only language from: {symbol_label}.",
            "No science terms yet.",
        ],
    )
    micro_check.setdefault(
        "prompt",
        "Using only your symbols, describe how things are set up right now."
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
        "- Use ONLY simple symbols and tokens from the learner's symbolic universe.\n"
        "- NO physics terminology.\n"
        "- NO formal theory, equations, or explanations.\n"
        "- Speak entirely in the learner's own shows, teams, characters, and objects.\n"
        "- Do not explain quantum physics yet; only map and move their symbols.\n"
        "- Do not introduce any examples, franchises, or characters not explicitly named by the learner.\n"
    )

def domain_symbolic_lock(state: CobraState) -> str:
    """
    V7 HARD SYMBOLIC LOCK — DOMAIN LEVEL
    Forces the model to stay strictly inside the learner's symbolic universe.
    """
    symbols = v7_get_symbol_universe(state) or []

    symbol_list = ", ".join(str(s) for s in symbols) if symbols else "NO SYMBOLS PROVIDED"

    return (
        "SYMBOLIC LOCK (V7 — NON-NEGOTIABLE):\n"
        f"- You may ONLY use the following symbols: {symbol_list}\n"
        "- You may NOT introduce new shows, teams, people, metaphors, or examples.\n"
        "- Do NOT explain concepts.\n"
        "- Do NOT teach.\n"
        "- Do NOT generalize.\n"
        "- Treat symbols as tokens to be arranged, not stories to be told.\n"
        "- If a symbol is not listed above, it does not exist.\n"
    )

def micro_check_symbolic_lock() -> str:
    """
    V7 HARD MICRO-CHECK LOCK.
    This overrides all teaching instincts.
    """
    return (
        "MICRO-CHECK CONSTRAINTS (V7 — STRICT):\n"
        "- Respond using ONLY symbols, characters, teams, or objects from the learner’s symbolic universe.\n"
        "- This is NOT an explanation.\n"
        "- DO NOT define, describe, justify, or teach.\n"
        "- DO NOT generalize or abstract.\n"
        "- NO theory, science, or causal language.\n"
        "- One sentence only.\n"
        "- Treat this as verification, not instruction.\n"
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

    # NEW: Phase 2 stress-test flags
    if "phase2_stress_test" in payload:
        normalized["phase2_stress_test"] = payload["phase2_stress_test"]
    if "phase2_stress_mode" in payload:
        normalized["phase2_stress_mode"] = payload["phase2_stress_mode"]

    # Use normalized payload from this point on
    payload = normalized

    # Optional: allow caller to explicitly reseed the symbol universe
    reset_symbols = bool(payload.get("reset_symbols"))

    # V7 HARD RULE: client may never control domain or phase
    payload.pop("domain", None)
    payload.pop("phase", None)

    # Phase 2 stress flags (now read from normalized)
    phase2_stress_test = bool(payload.get("phase2_stress_test"))
    phase2_stress_mode = payload.get("phase2_stress_mode")

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
        # V7 HARD LOCK — DOMAIN 0 IS WRITE-ONCE (IMMUTABLE)
        # =====================================================
        if getattr(state, "domain0_complete", False):
            # Strip any attempt to reseed Domain 0 after lock
            payload.pop("want_to_understand", None)
            payload.pop("likes", None)
            payload.pop("interaction_mode", None)

        # =====================================================
        # V7 DOMAIN 0 — EXPLICIT SERVER CONFIRMATION (DETERMINISTIC)
        # =====================================================
        if not getattr(state, "domain0_complete", False):

            want = payload.get("want_to_understand")
            likes = payload.get("likes")
            mode = payload.get("interaction_mode")

            if isinstance(likes, str):
                likes = [s.strip() for s in likes.split(",") if s.strip()]

            domain0_ready = bool(
                want and
                isinstance(likes, list) and len(likes) > 0 and
                mode in ("learn", "adventure", "mastery")
            )

            if domain0_ready:
                confirmed = {
                    "want_to_understand": want,
                    "likes": likes,
                    "interaction_mode": mode
                }
            else:
                confirmed = None

            if not domain0_ready:
                return {
                    "domain": "D0",
                    "phase": "PHASE_1",
                    "intent": "REPAIR",
                    "repair_required": True,
                    "stability_assessment": "UNSTABLE",
                    "text": (
                        "Domain 0 incomplete. You must provide:\n"
                        "- what you want to understand\n"
                        "- what you naturally understand or like\n"
                        "- an interaction mode"
                    ),
                    "symbols_used": [],
                    "symbol_universe": [],
                    "state": {
                        "domain0_complete": False,
                        "domain0b_complete": False,
                        "phase2_active": False,
                    },
                    "next_domain_recommendation": "D0",
                    "media_suggestions": [],
                    "payload": {},
                    "micro_check": {
                        "prompt": "Answer the questions above.",
                        "expected_response_type": "conceptual",
                    },
                }

            # ONLY REACHED IF domain0_ready == True
            state.symbolic_universe = {
                "symbol_universe": confirmed,
                "domain0_raw": confirmed,
            }

                save_session_state(session_id, state)

                return {
                    "domain": "D0",
                    "phase": "PHASE_1",
                    "intent": "CONFIRMATION",
                    "introduced_new_symbols": False,
                    "repair_required": False,
                    "stability_assessment": "STABLE",
                    "text": "Symbolic universe confirmed. Domain 0 locked.",
                    "symbols_used": confirmed,
                    "symbol_universe": confirmed,
                    "state": {
                        "domain0_complete": True,
                        "domain0b_complete": bool(getattr(state, "domain0b_complete", False)),
                        "phase2_active": bool(getattr(state, "phase2_active", False)),
                    },
                    "next_domain_recommendation": "D0B",
                    "media_suggestions": [],
                    "payload": {},
                    "micro_check": {
                        "prompt": "Proceeding to Domain 0B.",
                        "expected_response_type": "conceptual",
                    },
                }

        # =====================================================
        # V7 INVARIANT (MANDATORY): symbolic_universe MUST be dict
        # app.py uses .setdefault(...) on symbolic_universe
        # =====================================================
        if getattr(state, "symbolic_universe", None) is None:
            state.symbolic_universe = {}
        elif isinstance(state.symbolic_universe, list):
            state.symbolic_universe = {"symbol_universe": state.symbolic_universe}
        elif not isinstance(state.symbolic_universe, dict):
            raise TypeError(f"Invalid symbolic_universe type: {type(state.symbolic_universe)}")

        state.symbolic_universe.setdefault("symbol_universe", [])
        state.symbolic_universe.setdefault("domain0_raw", [])

        # If caller wants to reseed symbols, clear the old universe
        if reset_symbols:
            state.symbolic_universe = {"symbol_universe": [], "domain0_raw": []}
            state.domain0_complete = False  # only if you want to re-run Domain 0

        # =====================================================
        # V7 HARD GATE — MICRO-CHECK REQUIRES SYMBOL UNIVERSE
        # =====================================================
        if payload.get("micro_response"):
            if not getattr(state, "domain0_complete", False):
                return {
                    "domain": "D0",
                    "intent": "REPAIR",
                    "text": "Symbol universe is empty. Domain 0 must be completed before micro-check.",
                    "repair_required": True,
                    "symbols_used": [],
                    "symbol_universe": [],
                    "state": {
                        "domain0_complete": False,
                        "domain0b_complete": bool(getattr(state, "domain0b_complete", False)),
                },
             }

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
            payload.setdefault("prompt", "")

        # =====================================================
        # V7 HARD GATE — DOMAIN 0B REQUIRED BEFORE DOMAIN 1
        # =====================================================
        if v7_requires_domain0b(state):
            response = v7_domain0b_response(state)
            save_session_state(session_id, state)
            return response

        # ---------------------------
        # Build prompt
        # ---------------------------
        prompt = payload.get("prompt", "")

        # If a specific Phase 2 stress-test mode is selected, annotate the prompt
        if phase2_stress_mode:
            prompt = f"Phase 2 stress-test mode: {phase2_stress_mode}.\n\n{prompt}"

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
        # V7 DOMAIN 0B — RECORD INTERACTION CONSTRAINTS
        # =====================================================
        if v7_requires_domain0b(state) and payload.get("micro_response"):
            v7_record_domain0b_answer(
                state=state,
                response=payload["micro_response"]
            )
            save_session_state(session_id, state)

        # =====================================================
        # V7 HARD GUARD — prevent Domain 0 / 0B reseeding via prompt
        # =====================================================
        if state.domain0_complete:
            # Prompt may continue conversation, but not reseed symbols
            pass

        ## ---------------------------
        # Call V7 engine
        # ---------------------------
        
        # V7 HARD OVERRIDE — Domain 0 always runs until complete
        if not getattr(state, "domain0_complete", False):
            expected_domain = Domain.D0
        elif v7_requires_domain0b(state):
            expected_domain = Domain.D0B
        else:
            expected_domain = server_expected_domain(state)

        # -------------------------------------------------
        # PHASE 1 TRANSFER GATE (V7)
        # -------------------------------------------------

        if (
            getattr(state, "domain0_complete", False)
            and not getattr(state, "phase2_active", False)
            and v7_phase1_transfer_required(state)
            and expected_domain == Domain.D5
        ):
            # Surface the transfer check prompt instead of advancing
            response = v7_phase1_transfer_response(state)
            save_session_state(session_id, state)
            return response

        # PHASE 2 STRESS-TEST GATE (Fix #9)
        if (
            getattr(state, "phase2_active", False)
            and phase2_stress_test
            and v7_phase2_stress_test_required(state)
            and not phase2_stress_mode
        ):
            response = v7_phase2_stress_test_prompt(state)
            save_session_state(session_id, state)
            return response

        # OPTIONAL: PHASE 2 EXPANSION INVITE (V7)
        if (
            getattr(state, "phase2_active", False)
            and getattr(state, "phase1_transfer_complete", False)
            and not phase2_stress_test
            and not payload.get("expansion_opt_out")
        ):
            response = v7_expansion_prompt(state)
            save_session_state(session_id, state)
            return response

        expected_phase = "PHASE_2" if getattr(state, "phase2_active", False) else "PHASE_1"
        symbol_universe = server_symbol_universe(payload, state) or []

        # V7 HARD RULE:
        # Domain 0 completes IFF symbol universe is committed
        if not getattr(state, "domain0_complete", False):
            if isinstance(symbol_universe, list) and symbol_universe:
                state.symbolic_universe["symbol_universe"] = symbol_universe
                state.domain0_complete = True

        # =====================================================
        # V7 SYMBOLIC SCOPE LOCK — GLOBAL (ALL DOMAINS)
        # =====================================================

        symbol_scope_instruction = (
            "SYMBOLIC SCOPE RULE (V7 — HARD CONSTRAINT):\n"
            "- You may ONLY use symbols explicitly provided in the symbol_universe.\n"
            "- Do NOT invent new characters, teams, shows, metaphors, examples, or analogies.\n"
            "- Do NOT generalize or substitute with similar symbols.\n"
            "- If something cannot be expressed using the given symbols, say so explicitly.\n"
        )

        prompt = symbol_scope_instruction + "\n" + prompt

        # A2: Enforce 'Domain 1 = NO THEORY'
        if expected_domain == Domain.D1:
            prompt = (
                domain1_style_instruction()
                + "\n\n"
                + domain_symbolic_lock(state)
                + "\n\n"
                + prompt
            )

        response = call_model_with_retry_v7(
            prompt=prompt,
            state=state,
            expected_domain=expected_domain,
            expected_phase=expected_phase,
            symbol_universe=symbol_universe,
        )
        # =====================================================
        #  FIX 4 — HARD RESPONSE ENVELOPE FREEZE (V7)
        # =====================================================

        # 1. response MUST be a dict
        if not isinstance(response, dict):
            raise TypeError(
                f"V7 violation: response envelope corrupted: {type(response)}"
            )

        # 2. Freeze reference (prevent accidental reassignment)
        response = dict(response)

        # 3. Enforce payload shape
        payload_obj = response.get("payload")
        if payload_obj is None:
            payload_obj = {}
        elif not isinstance(payload_obj, dict):
            raise TypeError(
                f"V7 violation: payload must be dict, got {type(payload_obj)}"
            )
        response["payload"] = payload_obj

        # 4. Enforce blocks isolation
        blocks = payload_obj.get("blocks")
        if blocks is not None and not isinstance(blocks, list):
            raise TypeError(
                f"V7 violation: payload.blocks must be list, got {type(blocks)}"
            )

        # 5. Enforce micro_check isolation
        micro = response.get("micro_check")
        if micro is None:
            response["micro_check"] = {}
        elif not isinstance(micro, dict):
            raise TypeError(
            f"V7 violation: micro_check must be dict, got {type(micro)}"
        )   

        # 6. Enforce text scalar
        if not isinstance(response.get("text", ""), str):
            response["text"] = str(response.get("text", ""))

        # =====================================================
        # V7 HARD RESPONSE NORMALIZATION (ROBUST)
        # =====================================================
        if not isinstance(response, dict):
            raise TypeError(
                f"V7 violation: model returned {type(response)} instead of dict"
            )

        # Guarantee core shape early
        response.setdefault("payload", {})
        response.setdefault("micro_check", {})
        response.setdefault("text", "")

        # =====================================================
        # V7 HARD RESPONSE INVARIANT
        # =====================================================
        if not isinstance(response.get("payload"), dict):
            raise TypeError(
                f"V7 violation: response['payload'] must be dict, "
                f"got {type(response.get('payload'))}"
        )

        # =====================================================
        # V7 POST-CONDITION — DOMAIN 0 INTEGRITY (HARD)
        # =====================================================

        if getattr(state, "domain0_complete", False):
            symbols = v7_get_symbol_universe(state)

            if not isinstance(symbols, list) or len(symbols) == 0:
                # This state is ILLEGAL in V7 → force repair
                state.domain0_complete = False
                state.symbolic_universe = None

            save_session_state(session_id, state)

            return {
                "domain": "D0",
                "intent": "REPAIR",
                "repair_required": True,
                "text": (
                    "Domain 0 integrity failure. "
                    "A non-empty symbol universe is required before continuing."
                ),
                "symbol_universe": [],
                "symbols_used": [],
                "state": {
                    "domain0_complete": False,
                    "domain0b_complete": False,
                },
            }

        response = enforce_symbol_scope(response, state)

        # =====================================================
        # V7 GUARANTEE — symbols_used must never be empty
        # =====================================================
        symbols = v7_get_symbol_universe(state)
        if isinstance(symbols, list) and symbols:
            if not response.get("symbols_used"):
                response["symbols_used"] = symbols

        # PHASE 1 TRANSFER COMPLETION (simple heuristic)
        if (
            response.get("domain") == "D5"
            and response.get("stability_assessment") == "STABLE"
            and response.get("intent") == "TRANSFER_CHECK"
            and not getattr(state, "phase1_transfer_complete", False)
        ):
            state.phase1_transfer_complete = True

        # If server expects D1 but model still said D0, coerce to D1
        log_interaction(payload, response)

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
            # allow images during the first D1 micro-check
        ):
            # 1) Normalize Domain 1 layout + micro_check
            response = ensure_domain1_structure(response, state)

            # 2) Domain 1: symbolic image from user's symbols
            response = add_domain1_image_row_from_symbols(response, state)

            # 3) Domain 1: explanation text from user's symbols
            response = add_domain1_explanation_from_symbols(response, state)

            # 4) Domain 2: metaphoric images from user's symbol universe
            response = add_domain2_images_from_symbols(response, state)

            # 5) Domain 3: simple diagram from user's symbols
            response = add_domain3_diagram_from_symbols(response, state)

            mc = response.get("micro_check") or {}

            mc["required"] = True

            mc["rules"] = [
                "One sentence.",
                "Use only the learner’s own symbols.",
                "No theory. No explanation. No abstraction.",
            ]

            mc["prompt"] = "Using only your symbols, describe how things are set up right now."

            mc["system_constraint"] = micro_check_symbolic_lock()

            response["micro_check"] = mc

            response["intent"] = "MICRO_CHECK"
            state.domain1_microcheck_shown = True
            state.last_microcheck_response = copy.deepcopy(response)
            state.awaiting_micro_check = True
        # =====================================================
        # Always enrich D1/D2/D3 with symbol-based visuals/text
        # =====================================================
        dom = response.get("domain")

        if dom == "D1":
            response = add_domain1_image_row_from_symbols(response, state)
            response = add_domain1_explanation_from_symbols(response, state)

        if response.get("domain") == "D1":
            response["_symbolic_blocks_locked"] = True

        elif dom == "D2":
            response = add_domain2_images_from_symbols(response, state)

        elif dom == "D2B":
            response = add_domain2b_explanation_from_symbols(response, state)

        elif dom == "D3":
            response = add_domain3_diagram_from_symbols(response, state)
        
        elif dom == "D3B":
            response = add_domain3b_explanation_from_symbols(response, state)
        
        elif dom == "D4":
            response = add_domain4_explanation_from_symbols(response, state)
        
        elif dom == "D5":
            response = add_domain5_explanation_from_symbols(response, state)

        # =====================================================
        # V7 MICRO-CHECK ARMING (SERVER-OWNED)
        # =====================================================
        
        if response.get("intent") == "MICRO_CHECK":
            state.last_microcheck_response = copy.deepcopy(response)
            state.awaiting_micro_check = True
        
        # -------------------------------------------------
        # V7 UI CONTRACT: surface server-owned state
        # -------------------------------------------------
        response["state"] = {
            "domain0_complete": bool(getattr(state, "domain0_complete", False)),
            "domain0b_complete": bool(getattr(state, "domain0b_complete", False)),
            "phase2_active": bool(getattr(state, "phase2_active", False)),
            "phase2_stress_test": bool(payload.get("phase2_stress_test")),
            "phase2_stress_mode": payload.get("phase2_stress_mode"),
        }

        symbols = v7_get_symbol_universe(state)
        if isinstance(symbols, list):
            response["symbol_universe"] = symbols

        intro_lines = []

        # -------------------------------------------------
        # NEW: Domain 1 intro preface on first D1 turn
        # -------------------------------------------------
        if (
            response.get("domain") == "D1"
            and not getattr(state, "domain1_intro_shown", False)
        ):
            # Determine whether Domain-1 content already exists
            has_blocks = bool(response.get("payload", {}).get("blocks"))
            has_text = bool((response.get("text") or "").strip())

        # Only inject intro text if NOTHING meaningful exists yet
            if not has_blocks and not has_text:

                symbolic_universe_label = " + ".join(
                    getattr(state, "symbol_universe_labels", [])
                ) if getattr(state, "symbol_universe_labels", None) else "your symbols"

                mode = getattr(state, "interaction_mode", None)
                mode_label = (
                    mode.value.capitalize()
                    if hasattr(mode, "value")
                    else str(mode) if mode else "Unknown"
                )

                intro_lines.extend([
                    "Domain 1 initialized.",
                    f"Symbolic universe = {symbolic_universe_label}",
                    f"Mode = {mode_label}",
                    "",
                    "We move bottom-up.",
                    "",
                ])

        if intro_lines:
            response["text"] = "\n".join(intro_lines)

        # Mark intro as shown regardless, so it never fires again
        state.domain1_intro_shown = True

        # C2: ensure response has stable shape (NO DOMAIN MUTATION)

        if "intent" not in response:
            response["intent"] = "NORMAL"

        # HARD CLEANUP
        payload = response.get("payload") or {}
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            payload["blocks"] = remove_placeholder_images(blocks)
            response["payload"] = payload

        # =====================================================
        # V7 UI CONTRACT — SURFACE SYMBOL UNIVERSE (AUTHORITATIVE)
        # =====================================================
        symbols = v7_get_symbol_universe(state)
        if isinstance(symbols, list):
            response["symbol_universe"] = symbols

        save_session_state(session_id, state)
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "domain": "SERVER_ERROR",
            "intent": "ERROR",
            "text": f"Internal Server Error: {str(e)}"
        }



