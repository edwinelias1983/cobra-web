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

    # Use the real session_id if present, otherwise fall back to a new one
    session_id = payload.get("session_id") or str(uuid.uuid4())
    state_json = response_json  # or whatever you actually want to store

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

def server_symbol_universe(payload: dict, state: CobraState):
    """
    Server-owned symbolic universe:

    - Before Domain 0 is complete OR when reset_symbols is true:
      accept from payload and commit into state.
    - Otherwise: ignore payload and use stored state only.
    """
    reset_symbols = bool(payload.get("reset_symbols"))

    # Allow payload to (re)seed when Domain 0 not locked OR reset is requested
    if (not getattr(state, "domain0_complete", False) or reset_symbols) \
            and "symbol_universe" in payload:
        su = payload["symbol_universe"]
        state.symbolic_universe = su
        return su

    # After Domain 0 is complete, ignore payload and use stored state
    if isinstance(state.symbolic_universe, dict):
        return (
            state.symbolic_universe.get("symbol_universe")
            or state.symbolic_universe.get("symbols")
        )
    if isinstance(state.symbolic_universe, list):
        return state.symbolic_universe
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

def ensure_domain1_structure(response: dict) -> dict:
    """
    Make sure Domain 1 responses have the structured payload and micro_check fields
    that the V7 UI expects, and inject the Domain 1 mapping text.
    """
    if response.get("domain") != "D1":
        return response

    # 1) Ensure DOMAIN 1 symbolic visual block (what you already had)
    payload = response.get("payload") or {}

    blocks = payload.get("blocks")

    if blocks is None:
    # Keep existing structure, but do NOT assume shows / teams / franchises
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

    payload["blocks"] = blocks
    response["payload"] = payload

    # 2) Append “Quantum physics — stripped…” mapping into response["text"]
    text = (response.get("text") or "").strip()
    mapping_header = "Quantum physics — stripped of jargon:"

    # Avoid duplicating the mapping if the model ever echoes it
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
        text = text + "\n\n" + mapping_block if text else mapping_block
        response["text"] = text

    # 3) Ensure Domain 1 micro-check scaffold (unchanged)
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
        'Using only your symbols, describe how things are set up right now.'
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

        # If caller wants to reseed symbols, clear the old universe
        if reset_symbols:
            state.symbolic_universe = None
            state.domain0_complete = False  # only if you want to re-run Domain 0

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
        # V7 HARD GUARD — prevent Domain 0 / 0B reseeding via prompt
        # =====================================================
        if state.domain0_complete:
            # Prompt may continue conversation, but not reseed symbols
            pass

        ## ---------------------------
        # Call V7 engine
        # ---------------------------
        
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

        response = enforce_symbol_scope(response, state)

        # PHASE 1 TRANSFER COMPLETION (simple heuristic)
        if (
            response.get("domain") == "D5"
            and response.get("stability_assessment") == "STABLE"
            and response.get("intent") == "TRANSFER_CHECK"
            and not getattr(state, "phase1_transfer_complete", False)
        ):
            state.phase1_transfer_complete = True

        # If server expects D1 but model still said D0, coerce to D1
        if getattr(state, "domain0_complete", False) and response.get("domain") == "D0":
            response["domain"] = "D1"

        log_interaction(payload, response)

        # A4: Commit symbolic universe to state when Domain 0 completes
        if not getattr(state, "domain0_complete", False) and response.get("domain") == "D0":
            su = payload.get("symbol_universe")

            # If client didn't send a symbol_universe, derive it from likes
            if su is None:
                likes = payload.get("likes") or []
                if isinstance(likes, str):
                    likes = [likes]
                su = likes

            if su:
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
            # allow images during the first D1 micro-check
        ):
            # 1) Normalize Domain 1 layout + micro_check
            response = ensure_domain1_structure(response)

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
            state.last_microcheck_response = response
            state.awaiting_micro_check = True
        # =====================================================
        # Always enrich D1/D2/D3 with symbol-based visuals/text
        # =====================================================
        dom = response.get("domain")

        if dom == "D1":
            response = add_domain1_image_row_from_symbols(response, state)
            response = add_domain1_explanation_from_symbols(response, state)

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
            state.last_microcheck_response = response
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
            response["text"] = intro_text

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

        # HARD CLEANUP
        payload = response.get("payload") or {}
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            payload["blocks"] = remove_placeholder_images(blocks)
            response["payload"] = payload

        save_session_state(session_id, state)
        return response

    except Exception as e:
        return {
            "domain": "SERVER_ERROR",
            "intent": "ERROR",
            "text": f"Internal Server Error: {str(e)}"
        }

# FORCE_COMMIT
