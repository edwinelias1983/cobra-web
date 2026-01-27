# v7_domains.py
# COBRA V7 — Domain Contracts + Prompt Compiler (Phase A)

from typing import Optional, List
import json

# ============================================================
# DOMAIN CONTRACTS — CANONICAL V7
# ============================================================

DOMAIN_CONTRACTS = {

    # --------------------------------------------------------
    # DOMAIN 0 — Personal Symbol Map
    # --------------------------------------------------------
    "D0": {
        "purpose": "personal symbol universe creation",
        "allowed": [
            "questions about interests",
            "preferences",
            "likes",
            "personal references",
            "cultural anchors"
        ],
        "forbidden": [
            "explanations",
            "teaching",
            "theory",
            "metaphor",
            "scenes",
            "systems language"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 0B — Auditory / Style Stabilization
    # --------------------------------------------------------
    "D0B": {
        "purpose": "stabilization, recall, and communication style alignment",
        "allowed": [
            "reflection",
            "what stuck",
            "how it was said",
            "style preference",
            "pace or rhythm descriptions"
        ],
        "forbidden": [
            "teaching",
            "formal explanation",
            "new metaphors",
            "new symbols"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 1 — Symbolic (Primary Visual Layer)
    # --------------------------------------------------------
    "D1": {
        "purpose": "symbolic encoding using tokens from the learner’s world",
        "allowed": [
            "tokens",
            "short labels",
            "bullet symbols",
            "arrows",
            "iconic references"
        ],
        "forbidden": [
            "stories",
            "scenes",
            "narrative metaphor",
            "abstract systems language",
            "theory",
            "explanation"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 2 — Metaphoric (Second Visual Layer)
    # --------------------------------------------------------
    "D2": {
        "purpose": "metaphoric grounding via scenes or worlds",
        "allowed": [
            "scene",
            "world",
            "embodied metaphor",
            "story",
            "imagined environment"
        ],
        "forbidden": [
            "tokens",
            "arrows",
            "bullet abstraction",
            "formal systems language",
            "equations"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 2B — Analogical Structure Mapping
    # --------------------------------------------------------
    "D2B": {
        "purpose": "explicit mapping between two symbolic situations",
        "allowed": [
            "role alignment",
            "structural correspondence",
            "analogical mapping",
            "one-to-one comparison"
        ],
        "forbidden": [
            "new stories",
            "new symbols",
            "formal theory",
            "definitions"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 3 — Pattern
    # --------------------------------------------------------
    "D3": {
        "purpose": "pattern recognition and relational structure",
        "allowed": [
            "patterns",
            "relationships",
            "repetition",
            "variation",
            "simple diagrams"
        ],
        "forbidden": [
            "narrative storytelling",
            "metaphoric scenes",
            "formal theory",
            "symbol invention"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 3B — Temporal + Context Layer
    # --------------------------------------------------------
    "D3B": {
        "purpose": "temporal sequencing and contextual change",
        "allowed": [
            "before-and-after",
            "timeline",
            "progression",
            "situational shifts"
        ],
        "forbidden": [
            "abstract systems theory",
            "new metaphors",
            "equations",
            "definitions"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 4 — Systemic
    # --------------------------------------------------------
    "D4": {
        "purpose": "system-level constraints and interactions",
        "allowed": [
            "constraints",
            "limits",
            "feedback",
            "structure",
            "rules of the system"
        ],
        "forbidden": [
            "new symbols",
            "narrative scenes",
            "storytelling",
            "personal examples"
        ],
    },

    # --------------------------------------------------------
    # DOMAIN 5 — Paradox / Tension
    # --------------------------------------------------------
    "D5": {
        "purpose": "live tension between incompatible truths",
        "allowed": [
            "paradox",
            "tension",
            "contradiction",
            "coexisting truths"
        ],
        "forbidden": [
            "resolution",
            "simplification",
            "teaching",
            "step-by-step explanation"
        ],
    },
}

# ============================================================
# PROMPT COMPILER — PHASE A
# ============================================================

def compile_domain_prompt(
    domain: str,
    user_prompt: str,
    symbolic_universe: Optional[List[str]] = None
) -> str:
    """
    Phase A compiler:
    - Injects domain intent and constraints
    - Injects symbolic universe lock
    - Does NOT validate output (Phase B)
    """

    contract = DOMAIN_CONTRACTS.get(domain)

    # Fail open if domain is unknown (Phase A safety)
    if not contract:
        return user_prompt

    universe_block = ""
    if symbolic_universe:
        universe_block = (
            "SYMBOLIC UNIVERSE (LOCKED):\n"
            f"- {', '.join(symbolic_universe)}\n\n"
            "RULE:\n"
            "- You may ONLY use the symbols listed above.\n"
            "- Do NOT introduce foreign metaphors, examples, or references.\n"
        )

    system_instructions = f"""
COBRA V7 ACTIVE

CURRENT DOMAIN: {domain}
DOMAIN PURPOSE:
- {contract['purpose']}

ALLOWED OUTPUT FORMS:
- {', '.join(contract['allowed'])}

FORBIDDEN OUTPUT FORMS:
- {', '.join(contract['forbidden'])}

Violation of these constraints is an error.
{universe_block}
"""

    return (
        system_instructions.strip()
        + "\n\nUSER PROMPT:\n"
        + user_prompt.strip()
    )
# ============================================================
# PHASE B — DOMAIN OUTPUT VALIDATION (V7)
# ============================================================

class DomainViolation(Exception):
    pass


def validate_domain_output(domain: str, response: dict):
    """
    Phase-B validator.
    Enforces domain contracts on model output.
    """

    contract = DOMAIN_CONTRACTS.get(domain)
    if not contract:
        return  # fail-open for unknown domains (temporary)

    text = (response.get("text") or "").lower()
    payload = response.get("payload") or {}

    violations = []

    # --- Forbidden content heuristics (minimal, explicit) ---
    for forbidden in contract["forbidden"]:
        if forbidden.lower() in text:
            violations.append(f"Forbidden form detected: {forbidden}")

    # --- Domain-specific hard rules ---
    if domain == "D1":
        # No stories, scenes, or narrative verbs
        banned_markers = ["story", "scene", "once upon", "imagine", "narrative"]
        if any(b in text for b in banned_markers):
            violations.append("Narrative content detected in Domain 1")

    if domain == "D2":
        # No arrows / bullet abstraction
        if "→" in text or "-" in text:
            violations.append("Abstract tokenization detected in Domain 2")

    # --- Symbol usage must be declared ---
    if not response.get("symbols_used"):
        violations.append("symbols_used missing or empty")

    if violations:
        raise DomainViolation({
            "domain": domain,
            "violations": violations,
            "raw_text": response.get("text", "")
        })
def domain_repair_prompt(domain: str, violation_obj: dict) -> str:
    return f"""
REPAIR MODE — COBRA V7

The previous response violated Domain {domain} rules.

Violations:
{json.dumps(violation_obj, indent=2)}

You must regenerate the response so that:
- It strictly follows Domain {domain} constraints
- It uses ONLY the allowed output forms
- It introduces NO new symbols
- It does NOT explain the mistake

Return ONLY a corrected response.
"""
