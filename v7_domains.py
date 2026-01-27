# v7_domains.py
# COBRA V7 — Domain Contracts + Prompt Compiler (Phase A)

DOMAIN_CONTRACTS = {
    "D0": {
        "purpose": "symbol universe creation",
        "allowed": [
            "questions",
            "preferences",
            "likes",
            "personal references"
        ],
        "forbidden": [
            "explanations",
            "theory",
            "metaphor",
            "scenes"
        ]
    },

    "D0B": {
        "purpose": "stabilization and recall",
        "allowed": [
            "reflection",
            "concrete recall",
            "what stuck",
            "how it was said"
        ],
        "forbidden": [
            "teaching",
            "formal explanation",
            "new metaphors"
        ]
    },

    "D1": {
        "purpose": "symbolic encoding",
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
            "abstract systems language"
        ]
    },

    "D2": {
        "purpose": "metaphoric grounding",
        "allowed": [
            "scene",
            "world",
            "embodied metaphor",
            "story"
        ],
        "forbidden": [
            "tokens",
            "arrows",
            "bullet abstraction",
            "formal systems language"
        ]
    }
}


def compile_domain_prompt(domain: str, user_prompt: str, symbolic_universe: list[str] | None):
    """
    Phase A compiler:
    - Enforces domain intent via instruction
    - Does NOT yet validate output (Phase B)
    """

    contract = DOMAIN_CONTRACTS.get(domain)

    if not contract:
        return user_prompt  # fail open for unknown domains (temporary)

    universe_block = ""
    if symbolic_universe:
        universe_block = (
            "You MUST use ONLY the following symbolic universe:\n"
            f"{', '.join(symbolic_universe)}\n"
            "Do not introduce foreign metaphors or examples.\n"
        )

    system_instructions = f"""
COBRA V7 ACTIVE
Current Domain: {domain}
Domain Purpose: {contract['purpose']}

ALLOWED OUTPUT FORMS:
- {', '.join(contract['allowed'])}

FORBIDDEN OUTPUT FORMS:
- {', '.join(contract['forbidden'])}

Violation of these constraints is an error.
{universe_block}
"""

    return system_instructions.strip() + "\n\nUSER PROMPT:\n" + user_prompt