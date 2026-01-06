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

def v7_canonical_domain(domain_value: str) -> str:
    return V7_DOMAIN_CANONICAL_MAP.get(domain_value, domain_value)

def v7_enforce_domain_progression(state: CobraState, expected_domain: str):
    """
    expected_domain here should be a V7 label (D0/D0B/D1/...)
    This guard uses state.current_domain (your enum values) and maps it to V7 labels.
    """
    current = v7_canonical_domain(state.current_domain.value if hasattr(state.current_domain, "value") else str(state.current_domain))
    if current not in V7_DOMAIN_SEQUENCE or expected_domain not in V7_DOMAIN_SEQUENCE:
        return
    ci = V7_DOMAIN_SEQUENCE.index(current)
    ni = V7_DOMAIN_SEQUENCE.index(expected_domain)
    if ni not in (ci, ci + 1):
        raise RuntimeError(f"[V7 VIOLATION] Illegal domain jump: {current} → {expected_domain}")

def v7_expected_microcheck_type(expected_domain: str) -> str:
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
    ]
    if any(p in t for p in forbidden):
        raise RuntimeError("[V7 VIOLATION] Generic fallback detected in text")

def validate_cobra_response_v7(
    raw_text: str,
    expected_domain: str,
    expected_phase: str,
    state: CobraState,
    symbol_universe=None
):
    """
    V7 wrapper: uses your existing validate_cobra_response, then adds V7 hard gates.
    No changes to the original function.
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

    # V7 ADD: micro_check representation lock
    mc = parsed.get("micro_check", {})
    got_type = mc.get("expected_response_type")
    want_type = v7_expected_microcheck_type(expected_domain)
    if got_type != want_type:
        return None, [f"[V7 VIOLATION] micro_check.expected_response_type must be '{want_type}' for domain {expected_domain} (got '{got_type}')"]

    # V7 ADD: hard block generic fallback
    try:
        v7_block_generic_fallback_text(parsed.get("text", ""))
    except RuntimeError as e:
        return None, [str(e)]

    # V7 ADD: summary gate (do not allow SUMMARY until D5)
    if parsed.get("intent") == "SUMMARY" and expected_domain != "D5":
        return None, ["[V7 VIOLATION] SUMMARY not allowed before D5"]

    return parsed, []

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
    No changes to your original call_model_with_retry.
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

    # Update state domain only after successful V7-validated response
    # (state.current_domain uses your enum values; we map expected_domain back if needed)
    # We do NOT mutate your enum; we only advance when caller advances expected_domain.
    return parsed
