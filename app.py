import json
import os
from jsonschema import Draft202012Validator
from openai import OpenAI

# =========================
# OpenAI Client
# =========================
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# Load COBRA Response Schema (ENFORCED)
# =========================
SCHEMA_PATH = "cobra_response_schema_v1.json"
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    COBRA_SCHEMA = json.load(f)

ALLOWED_MEDIA_DOMAINS = {"D1", "D2"}

RETRY_PROMPT_TEMPLATE = """You MUST return a single valid JSON object that matches the COBRA Response Schema V1 exactly.

The previous response FAILED validation for these reasons:
- {VALIDATION_ERRORS}

Fix the response by outputting ONLY JSON (no markdown, no commentary, no code fences).

Hard Rules:
1) Keep the same domain and phase: domain={EXPECTED_DOMAIN}, phase={EXPECTED_PHASE}.
2) introduced_new_symbols MUST be boolean false unless explicitly allowed.
3) intent MUST be one of: QUESTION, EXPLANATION, MICRO_CHECK, REPAIR, TRANSFER_CHECK, PHASE2_CHOICE, STRESS_TEST, EXPANSION_INVITE, SUMMARY
4) stability_assessment MUST be one of: STABLE, UNSTABLE, UNKNOWN
5) Include exactly ONE micro_check object with:
   - prompt
   - expected_response_type
6) media_suggestions MUST be an array (possibly empty) and only allowed if domain is D1 or D2.
7) If domain is D1 or D2, you MUST include:
   - symbol_universe (non-empty array)
   - symbols_used (array) AND every entry in symbols_used MUST be an element of symbol_universe
8) Return ONLY the corrected JSON object now.
"""

# =========================
# LLM CALL (JSON GUARDED)
# =========================
def llm_call(prompt: str, expected_domain: str, expected_phase: str) -> str:
    schema_str = json.dumps(COBRA_SCHEMA, ensure_ascii=False)

    system = (
    "You are operating under the COBRA protocol. "
    "You MUST output ONLY a single valid JSON object that validates against the provided schema. "
    f"Required: domain must equal {expected_domain}; phase must equal {expected_phase}. "
    "DO NOT explain the system, domain, phase, or protocol. "
    "DO NOT describe what you are doing. "
    "DO NOT ask meta-questions about understanding the explanation. "
    "Respond ONLY to the user’s content using their language and lived examples. "
    "Do not include markdown, explanations, or code fences.\n\n"
    "COBRA Response Schema V1 (authoritative):\n"
    + schema_str
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content

    # ---------- JSON GUARD (NEW) ----------
    try:
        json.loads(raw)
    except Exception:
        # Force retry path instead of crashing
        return json.dumps({
            "domain": expected_domain,
            "phase": expected_phase,
            "intent": "REPAIR",
            "introduced_new_symbols": False,
            "repair_required": True,
            "stability_assessment": "UNSTABLE",
            "text": raw,
            "micro_check": {
                "prompt": "Please restate your response in valid JSON only.",
                "expected_response_type": "json"
            }
        })

    return raw


# =========================
# VALIDATION
# =========================
def validate_cobra_response(
    raw_text: str,
    expected_domain: str,
    expected_phase: str,
    symbol_universe=None
):
    errors = []

    try:
        data = json.loads(raw_text)
    except Exception as e:
        return None, [f"Invalid JSON: {e}"]

    validator = Draft202012Validator(COBRA_SCHEMA)
    for err in sorted(validator.iter_errors(data), key=lambda e: e.path):
        path = ".".join([str(p) for p in err.path]) or "(root)"
        errors.append(f"Schema error at {path}: {err.message}")

    if data.get("domain") != expected_domain:
        errors.append(f"Wrong domain: got {data.get('domain')} expected {expected_domain}")

    if data.get("phase") != expected_phase:
        errors.append(f"Wrong phase: got {data.get('phase')} expected {expected_phase}")

    if expected_domain in ("D1", "D2") and isinstance(symbol_universe, list):
        used = data.get("symbols_used", [])

        if not isinstance(used, list):
            errors.append("symbols_used must be an array")
        else:
            invalid_used = [s for s in used if s not in symbol_universe]
            if invalid_used:
                errors.append(
                    f"symbols_used contains items not in expected symbol_universe: {invalid_used}"
                )
                data["intent"] = "REPAIR"
                data["repair_required"] = True

        data["symbol_universe"] = symbol_universe

    if data.get("introduced_new_symbols") is True:
        errors.append("introduced_new_symbols must be false unless explicitly allowed")

    return (data if not errors else None), errors


# =========================
# RETRY WRAPPER
# =========================
def call_model_with_retry(
    prompt: str,
    expected_domain: str,
    expected_phase: str,
    symbol_universe=None,
    max_retries: int = 2,
    strict_schema: bool = False,
):
    raw = llm_call(prompt, expected_domain, expected_phase)
    parsed, errors = validate_cobra_response(
        raw,
        expected_domain,
        expected_phase,
        symbol_universe=symbol_universe,
    )

    attempts = 0
    while errors and attempts < max_retries:
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            VALIDATION_ERRORS="\n- ".join(errors),
            EXPECTED_DOMAIN=expected_domain,
            EXPECTED_PHASE=expected_phase,
        )

        raw = llm_call(retry_prompt, expected_domain, expected_phase)
        parsed, errors = validate_cobra_response(
            raw,
            expected_domain,
            expected_phase,
            symbol_universe=symbol_universe,
        )

        attempts += 1

    if errors:
        raise ValueError("Model failed validation after retries: " + "; ".join(errors))

    return parsed


# ============================================================
# V6 COBRA STATE + STRUCTURAL GATES (UNCHANGED)
# ============================================================

from enum import Enum
from dataclasses import dataclass, field

class InteractionMode(str, Enum):
    learn = "learn"
    adventure = "adventure"
    mastery = "mastery"

class Domain(str, Enum):
    D0 = "domain_0"
    D0B = "domain_0b"
    D1 = "domain_1"
    D2 = "domain_2"
    D2B = "domain_2b"
    D3 = "domain_3"
    D3B = "domain_3b"
    D4 = "domain_4"
    D5 = "domain_5"

@dataclass
class CobraState:
    interaction_mode: InteractionMode | None = None
    current_domain: Domain = Domain.D0
    stamina_used: bool = False
    consolidation_active: bool = False
    last_microcheck_passed: bool = False
    symbolic_universe: dict = field(default_factory=dict)
    auditory_universe: dict = field(default_factory=dict)

def maybe_offer_stamina_gate(state: CobraState) -> bool:
    if (
        state.last_microcheck_passed
        and not state.stamina_used
        and not state.consolidation_active
    ):
        state.stamina_used = True
        return True
    return False

def enter_consolidation(state: CobraState):
    state.consolidation_active = True

PRESENCE_MARKERS = [
    "We stay here.",
    "This is the right place to slow down.",
    "We’re not moving yet."
]

def maybe_add_presence_marker(state: CobraState, repair_event: bool) -> str | None:
    if repair_event and state.interaction_mode == InteractionMode.mastery:
        return PRESENCE_MARKERS[0]
    return None


# ============================================================
# V7 ADDITIONS (NO DELETIONS / NO REFACTOR)
# ============================================================

# V7 ADD: canonical domain sequence (protocol order)
V7_DOMAIN_SEQUENCE = ["D0", "D0B", "D1", "D2", "D2B", "D3", "D3B", "D4", "D5"]

# V7 ADD: map your existing enum values to V7 canonical labels
V7_DOMAIN_CANONICAL_MAP = {
    "domain_0": "D0",
    "domain_0b": "D0B",
    "domain_1": "D1",
    "domain_2": "D2",
    "domain_2b": "D2B",
    "domain_3": "D3",
    "domain_3b": "D3B",
    "domain_4": "D4",
    "domain_5": "D5",
}

# V7 ADD: reverse map (V7 label -> enum value string)
V7_DOMAIN_REVERSE_MAP = {v: k for k, v in V7_DOMAIN_CANONICAL_MAP.items()}

# V7 ADD: expected micro-check response type by domain (representation lock)
V7_MICROCHECK_TYPE_BY_DOMAIN = {
    "D1": "symbolic",
    "D2": "metaphoric",
    "D2B": "analogy",
    "D3": "pattern",
    "D3B": "temporal",
    "D4": "systemic",
    "D5": "paradox",
}

# V7 ADD: intents that should not allow advance unless explicitly set
V7_LOCKING_INTENTS = {"MICRO_CHECK", "REPAIR"}

def v7_canonical_domain(domain_value: str) -> str:
    return V7_DOMAIN_CANONICAL_MAP.get(domain_value, domain_value)

def v7_state_domain_label(state: CobraState) -> str:
    """
    Returns canonical V7 label for current state domain.
    """
    try:
        return v7_canonical_domain(state.current_domain.value)
    except Exception:
        return v7_canonical_domain(str(state.current_domain))

def v7_enforce_domain_progression(state: CobraState, expected_domain: str):
    """
    expected_domain should be V7 label (D0/D0B/D1/...)
    Enforces: stay or advance by exactly one domain step.
    """
    current = v7_state_domain_label(state)
    if current not in V7_DOMAIN_SEQUENCE or expected_domain not in V7_DOMAIN_SEQUENCE:
        return
    ci = V7_DOMAIN_SEQUENCE.index(current)
    ni = V7_DOMAIN_SEQUENCE.index(expected_domain)
    if ni not in (ci, ci + 1):
        raise RuntimeError(f"[V7 VIOLATION] Illegal domain jump: {current} → {expected_domain}")

def v7_expected_microcheck_type(expected_domain: str) -> str:
    if expected_domain in ("D0", "D0B"):
        return "conceptual"
    return V7_MICROCHECK_TYPE_BY_DOMAIN.get(expected_domain, "conceptual")

def v7_block_generic_fallback_text(text: str):
    """
    Hard stop if the model outputs definition-style fallback even if JSON/schema-valid.
    """
    t = (text or "").lower()
    forbidden = [
        "is a fundamental theory",
        "can be defined as",
        "refers to",
        "in simple terms",
        "is the study of",
        "a branch of",
    ]
    if any(p in t for p in forbidden):
        raise RuntimeError("[V7 VIOLATION] Generic fallback detected in text")

def v7_enforce_media_domain(parsed: dict):
    """
    media_suggestions must be array and only allowed for D1/D2.
    """
    dom = parsed.get("domain")
    media = parsed.get("media_suggestions", [])
    if media is None:
        return
    if not isinstance(media, list):
        raise RuntimeError("[V7 VIOLATION] media_suggestions must be an array")
    if media and dom not in ("D1", "D2"):
        raise RuntimeError("[V7 VIOLATION] media_suggestions only allowed in D1 or D2")

def v7_enforce_symbol_binding(parsed: dict, expected_domain: str, symbol_universe: list | None):
    """
    If in D1 or D2:
      - symbol_universe must exist and be non-empty
      - symbols_used must be list
      - symbols_used ⊆ symbol_universe
    """
    if expected_domain not in ("D1", "D2"):
        return

    su = parsed.get("symbol_universe", None)
    used = parsed.get("symbols_used", None)

    # prefer passed-in symbol_universe (controller truth)
    if isinstance(symbol_universe, list):
        su = symbol_universe

    if not isinstance(su, list) or len(su) == 0:
        raise RuntimeError("[V7 VIOLATION] symbol_universe must be non-empty in D1/D2")

    if not isinstance(used, list):
        raise RuntimeError("[V7 VIOLATION] symbols_used must be an array")

    # You may choose to require non-empty used. (V7 grounding)
    if len(used) == 0:
        raise RuntimeError("[V7 VIOLATION] symbols_used must be non-empty in D1/D2")

    invalid = [s for s in used if s not in su]
    if invalid:
        raise RuntimeError(f"[V7 VIOLATION] symbols_used contains items not in symbol_universe: {invalid}")

def v7_enforce_introduced_symbols(parsed: dict):
    """
    introduced_new_symbols must be false unless explicitly allowed upstream.
    Schema allows boolean; V7 enforcement rejects True.
    """
    if parsed.get("introduced_new_symbols") is True:
        raise RuntimeError("[V7 VIOLATION] introduced_new_symbols must be false (unless explicitly allowed)")

def v7_enforce_microcheck_type(parsed: dict, expected_domain: str):
    # V7 RULE: Domain 0 has NO micro-check type enforcement
    if expected_domain in ("D0", "D0B"):
        return

    mc = parsed.get("micro_check", {})
    got_type = mc.get("expected_response_type")
    want_type = V7_MICROCHECK_TYPE_BY_DOMAIN.get(expected_domain)

    if want_type and got_type != want_type:
        raise RuntimeError(
            f"[V7 VIOLATION] micro_check.expected_response_type must be '{want_type}' "
            f"for domain {expected_domain} (got '{got_type}')"
        )

    mc = parsed.get("micro_check", {})
    got_type = mc.get("expected_response_type")
    want_type = V7_MICROCHECK_TYPE_BY_DOMAIN.get(expected_domain)

    if want_type and got_type != want_type:
        raise RuntimeError(
            f"[V7 VIOLATION] micro_check.expected_response_type must be '{want_type}' for domain {expected_domain} (got '{got_type}')"
        )

def v7_enforce_summary_gate(parsed: dict, expected_domain: str):
    if parsed.get("intent") == "SUMMARY" and expected_domain != "D5":
        raise RuntimeError("[V7 VIOLATION] SUMMARY not allowed before D5")

def v7_enforce_locking(parsed: dict):
    """
    If intent is MICRO_CHECK or REPAIR, advance_allowed must be false when present.
    This supports your “stay here” behavior.
    """
    intent = parsed.get("intent")
    if intent in V7_LOCKING_INTENTS:
        if parsed.get("advance_allowed") is True:
            raise RuntimeError("[V7 VIOLATION] advance_allowed must be false during MICRO_CHECK/REPAIR")

def validate_cobra_response_v7(
    raw_text: str,
    expected_domain: str,
    expected_phase: str,
    state: CobraState,
    symbol_universe=None
):
    """
    V7 wrapper: uses your existing validate_cobra_response, then adds V7 hard gates.
    No changes to the original validate_cobra_response.
    """
    parsed, errors = validate_cobra_response(
        raw_text,
        expected_domain,
        expected_phase,
        symbol_universe=symbol_universe
    )

    # If schema/base validation already failed, return as-is.
    if errors or parsed is None:
        return None, errors

    try:
        v7_enforce_introduced_symbols(parsed)
        v7_enforce_media_domain(parsed)
        v7_enforce_symbol_binding(parsed, expected_domain, symbol_universe if isinstance(symbol_universe, list) else None)
        v7_enforce_microcheck_type(parsed, expected_domain)
        v7_enforce_summary_gate(parsed, expected_domain)
        v7_enforce_locking(parsed)
        v7_block_generic_fallback_text(parsed.get("text", ""))
    except RuntimeError as e:
        return None, [str(e)]

    return parsed, []

def v7_set_state_domain_after_success(state: CobraState, expected_domain: str):
    """
    Advance state.current_domain to match expected_domain (V7 label).
    This is the enforcement hook that actually makes progression real.
    """
    enum_value = V7_DOMAIN_REVERSE_MAP.get(expected_domain)
    if not enum_value:
        return
    try:
        state.current_domain = Domain(enum_value)
    except Exception:
        return

def call_model_with_retry_v7(
    prompt: str,
    expected_domain: str,
    expected_phase: str,
    state: CobraState,
    symbol_universe=None,
    max_retries: int = 2,
):
    """
    V7 wrapper: enforces domain progression via state + validates with V7 invariants.
    Uses llm_call + retry loop.
    """
    v7_enforce_domain_progression(state, expected_domain)

    raw = llm_call(prompt, expected_domain, expected_phase)
    parsed, errors = validate_cobra_response_v7(
        raw,
        expected_domain,
        expected_phase,
        state=state,
        symbol_universe=symbol_universe
    )

    attempts = 0
    while errors and attempts < max_retries:
        retry_prompt = RETRY_PROMPT_TEMPLATE.format(
            VALIDATION_ERRORS="\n- ".join(errors),
            EXPECTED_DOMAIN=expected_domain,
            EXPECTED_PHASE=expected_phase,
        )

        raw = llm_call(retry_prompt, expected_domain, expected_phase)
        parsed, errors = validate_cobra_response_v7(
            raw,
            expected_domain,
            expected_phase,
            state=state,
            symbol_universe=symbol_universe
        )

        attempts += 1

    if errors:
        raise ValueError("Model failed V7 validation after retries: " + "; ".join(errors))

    # V7 ENFORCEMENT: advance state only after success
    v7_set_state_domain_after_success(state, expected_domain)
    v7_apply_interaction_mode_constraints(state, parsed)
    
    if (
        parsed.get("stability_assessment") == "STABLE"
        and maybe_offer_stamina_gate(state)
    ):
        return v7_stamina_gate_response(state)

    if state.consolidation_active:
        return v7_consolidation_response(state)
        if (
        expected_phase == "PHASE_1"
        and v7_phase1_transfer_required(state)
        and parsed.get("stability_assessment") == "STABLE"
        and expected_domain == "D5"
    ):
        return v7_phase1_transfer_response(state)
         # V7 PHASE 2: TOP-DOWN INVERSION ACTIVATION
    if (
        expected_phase == "PHASE_2"
        and v7_phase2_inversion_required(state)
    ):
        response = v7_phase2_prompt(state)
        if response:
            return response
       # V7 PHASE 2: STRESS-TEST MODE ACTIVATION
    if (
        expected_phase == "PHASE_2"
        and v7_phase2_stress_test_required(state)
    ):
        return v7_phase2_stress_test_prompt(state)

    return parsed

# ============================================================
# V7 REQUIRED: DOMAIN 0 ENFORCEMENT (MANDATORY FIRST STEP)
# ============================================================

D0_Q1 = "What do you want to understand today?"
D0_Q2 = (
    "What do you naturally understand or like? "
    "(sports, shows, music, games, hobbies, cultural references, memes, skills, etc.)"
)

INTERACTION_MODE_PROMPT = (
    "Choose an interaction mode:\n"
    "• Learn — guided understanding with gentle repair\n"
    "• Adventure — exploratory movement with delayed correction\n"
    "• Mastery — strict verification, immediate repair, and transfer enforcement"
)

def v7_requires_domain0(state: CobraState) -> bool:
    return state.interaction_mode is None or not state.symbolic_universe

def v7_domain0_response() -> dict:
    return {
        "domain": "D0",
        "phase": "PHASE_1",
        "intent": "QUESTION",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "UNKNOWN",
        "text": (
            f"{D0_Q1}\n\n"
            f"{D0_Q2}\n\n"
            f"{INTERACTION_MODE_PROMPT}"
        ),
        "micro_check": {
            "prompt": "Answer the questions above.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }
# ============================================================
# V7 REQUIRED: DOMAIN 0B — AUDITORY SYMBOL MAP
# ============================================================

D0B_QUESTIONS = [
    "Whose voice is really easy for you to understand?",
    "Who explains things in a way that immediately makes sense to you?",
    "Can you tell me something they explained that really stuck — and how they said it?",
    (
        "Is there a cultural or regional way of speaking that feels natural to your brain "
        "(e.g., Mexico City Spanish, Caribbean Spanish, AAVE, Spanglish, rural English)?"
    ),
    (
        "When explanations work best for you, do you prefer the big picture first, "
        "the details first, or moving back and forth?"
    ),
]

def v7_requires_domain0b(state: CobraState) -> bool:
    return not state.auditory_universe

def v7_domain0b_response(state: CobraState) -> dict:
    asked = state.auditory_universe.get("_asked", 0)
    if asked >= len(D0B_QUESTIONS):
        return {}

    question = D0B_QUESTIONS[asked]

    return {
        "domain": "D0B",
        "phase": "PHASE_1",
        "intent": "QUESTION",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "UNKNOWN",
        "text": question,
        "micro_check": {
            "prompt": "Answer in your own words.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }

def v7_record_domain0b_answer(state: CobraState, answer: str):
    if "_asked" not in state.auditory_universe:
        state.auditory_universe["_asked"] = 0
        state.auditory_universe["responses"] = []

    state.auditory_universe["responses"].append(answer)
    state.auditory_universe["_asked"] += 1
# ============================================================
# V7 REQUIRED: STAMINA GATE + CONSOLIDATION MODE
# ============================================================

STAMINA_PROMPT = (
    "Before continuing, do you want to:\n"
    "• continue at this depth\n"
    "• pause and consolidate\n"
    "• stop here and resume later?"
)

def v7_stamina_gate_response(state: CobraState) -> dict:
    return {
        "domain": v7_state_domain_label(state),
        "phase": "PHASE_1",
        "intent": "PHASE2_CHOICE",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "STABLE",
        "text": STAMINA_PROMPT,
        "micro_check": {
            "prompt": "Choose one option.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }

def v7_consolidation_response(state: CobraState) -> dict:
    return {
        "domain": v7_state_domain_label(state),
        "phase": "PHASE_1",
        "intent": "SUMMARY",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "STABLE",
        "text": (
            "We are consolidating.\n\n"
            "What is stable:\n"
            "- Core structure established\n\n"
            "What remains open:\n"
            "- Further refinement or transfer"
        ),
        "micro_check": {
            "prompt": "Does this summary match your understanding?",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }
# ============================================================
# V7 REQUIRED: PHASE 1 TRANSFER / VERIFICATION GATE
# ============================================================

def v7_phase1_transfer_required(state: CobraState) -> bool:
    """
    Phase 1 is incomplete until transfer has been verified.
    """
    return not getattr(state, "phase1_transfer_complete", False)

def v7_phase1_transfer_response(state: CobraState) -> dict:
    return {
        "domain": v7_state_domain_label(state),
        "phase": "PHASE_1",
        "intent": "TRANSFER_CHECK",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "UNKNOWN",
        "text": (
            "Before we continue, show transfer:\n\n"
            "1) Explain this using YOUR symbols.\n"
            "2) Explain the same idea using formal or academic language.\n\n"
            "We will compare alignment before moving on."
        ),
        "micro_check": {
            "prompt": "Provide both explanations.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }
# ============================================================
# V7 REQUIRED: PHASE 2 TOP-DOWN INVERSION LOGIC
# ============================================================

PHASE2_DOMAIN_SEQUENCE = ["D4", "D3B", "D3", "D2B", "D2", "D1"]

def v7_phase2_inversion_required(state: CobraState) -> bool:
    """
    Phase 2 runs only after Phase 1 transfer is complete.
    """
    return getattr(state, "phase1_transfer_complete", False)

def v7_phase2_next_domain(state: CobraState) -> str | None:
    """
    Determines the next domain in top-down inversion order.
    """
    current = v7_state_domain_label(state)
    if current not in PHASE2_DOMAIN_SEQUENCE:
        return PHASE2_DOMAIN_SEQUENCE[0]

    idx = PHASE2_DOMAIN_SEQUENCE.index(current)
    if idx + 1 >= len(PHASE2_DOMAIN_SEQUENCE):
        return None

    return PHASE2_DOMAIN_SEQUENCE[idx + 1]

def v7_phase2_prompt(state: CobraState) -> dict:
    next_domain = v7_phase2_next_domain(state)
    if not next_domain:
        return {}

    return {
        "domain": next_domain,
        "phase": "PHASE_2",
        "intent": "EXPLANATION",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "UNKNOWN",
        "text": (
            "We are now mapping the formal concept back onto your existing understanding.\n\n"
            "Explain how this formal idea connects to the structures you already built."
        ),
        "micro_check": {
            "prompt": "Map the formal idea to your earlier symbols or metaphors.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }
# ============================================================
# V7 REQUIRED: PHASE 2 STRESS-TEST MODES
# ============================================================

STRESS_TEST_MODES = {
    "metaphor": "Explain the formal concept again using a metaphor from your symbolic universe.",
    "boundary": "Where does this concept stop working? What assumptions must hold?",
    "objection": "What is the strongest objection or critique of this idea?",
    "transfer": "Apply this concept in a different context. What stays the same?"
}

def v7_phase2_stress_test_required(state: CobraState) -> bool:
    return getattr(state, "phase2_active", False)

def v7_phase2_stress_test_prompt(state: CobraState) -> dict:
    options = "\n".join(
        [f"• {k}: {v}" for k, v in STRESS_TEST_MODES.items()]
    )

    return {
        "domain": v7_state_domain_label(state),
        "phase": "PHASE_2",
        "intent": "STRESS_TEST",
        "introduced_new_symbols": False,
        "repair_required": False,
        "stability_assessment": "UNKNOWN",
        "text": (
            "Choose ONE way to pressure-test your understanding:\n\n"
            f"{options}"
        ),
        "micro_check": {
            "prompt": "Select one option and respond accordingly.",
            "expected_response_type": "conceptual"
        },
        "media_suggestions": []
    }

# ============================================================
# V7 REQUIRED: INTERACTION MODE BEHAVIOR ENFORCEMENT
# ============================================================

def v7_apply_interaction_mode_constraints(
    state: CobraState,
    parsed_response: dict
):
    """
    Modifies enforcement behavior based on interaction_mode.
    This does NOT change domain order or truth conditions.
    """

    mode = state.interaction_mode
    intent = parsed_response.get("intent")

    # LEARN MODE
    if mode == InteractionMode.learn:
        # allow mild imprecision; do not hard-block on first instability
        if parsed_response.get("stability_assessment") == "UNSTABLE":
            parsed_response["repair_required"] = True
            parsed_response.setdefault("advance_allowed", False)

    # ADVENTURE MODE
    elif mode == InteractionMode.adventure:
        # allow temporary instability unless it violates logic
        if intent not in ("REPAIR", "MICRO_CHECK"):
            parsed_response.setdefault("advance_allowed", True)

    # MASTERY MODE
    elif mode == InteractionMode.mastery:
        # strict: block advancement on ANY instability
        if parsed_response.get("stability_assessment") != "STABLE":
            parsed_response["repair_required"] = True
            parsed_response["advance_allowed"] = False

