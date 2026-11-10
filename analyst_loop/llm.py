"""Optional LLM backend interface — validated, not trusted.

The core loop never needs a model: every number in a memo is computed by
the statistics modules. An LLM may only *draft* wording on top of computed
output, and the draft is checked before it is used:

1. the decision label must be one of SHIP / HOLD / ITERATE / NEEDS-HUMAN;
2. every quoted figure in the draft must be a verbatim substring of the
   computed output.

Anything that fails either check escalates to a human. The backend is
NEVER run against a live model in this repo: unit tests use a mocked
transport only, and the default transport is None (offline).
"""

from __future__ import annotations

import json
from typing import Protocol

from .governance import DECISIONS


class LLMTransport(Protocol):
    """A text-in, text-out draft provider (e.g. an OpenAI/Azure client)."""

    def draft(self, prompt: str) -> str:
        ...


class MockTransport:
    """A canned transport for tests and demos. Never calls a network."""

    def __init__(self, draft_text: str):
        self.draft_text = draft_text
        self.prompts_seen: list[str] = []

    def draft(self, prompt: str) -> str:
        self.prompts_seen.append(prompt)
        return self.draft_text


def validate_draft(draft_text: str, computed_text: str) -> dict:
    """Validate an LLM draft against computed output.

    Returns {"decision", "claims", "escalated", "reason"}.
    """
    try:
        draft = json.loads(draft_text)
    except (json.JSONDecodeError, TypeError):
        return {
            "decision": "NEEDS-HUMAN",
            "claims": [],
            "escalated": True,
            "reason": "LLM draft is not valid JSON.",
        }
    decision = draft.get("decision")
    if decision not in DECISIONS:
        return {
            "decision": "NEEDS-HUMAN",
            "claims": [],
            "escalated": True,
            "reason": f"LLM returned unknown decision label: {decision!r}.",
        }
    claims = draft.get("claims", [])
    for claim in claims:
        quote = claim.get("quote", "")
        if not isinstance(quote, str) or quote == "" or quote not in computed_text:
            return {
                "decision": "NEEDS-HUMAN",
                "claims": [],
                "escalated": True,
                "reason": f"LLM quote is not a verbatim substring of computed output: {quote!r}.",
            }
    return {
        "decision": decision,
        "claims": claims,
        "escalated": False,
        "reason": "",
    }


class ValidatedMemoGenerator:
    """Drafts memo wording via an LLM transport, then validates it."""

    def __init__(self, transport: LLMTransport | None = None):
        self.transport = transport

    def generate(self, computed: dict) -> dict:
        """Return the validated draft, or an escalation if invalid."""
        computed_text = json.dumps(computed, sort_keys=True)
        if self.transport is None:
            return {
                "decision": "NEEDS-HUMAN",
                "claims": [],
                "escalated": True,
                "reason": "No LLM transport configured; core runs offline.",
            }
        prompt = (
            "Draft a decision memo as JSON with keys 'decision' and 'claims'. "
            "'decision' must be one of SHIP, HOLD, ITERATE, NEEDS-HUMAN. "
            "Each claim needs 'text' and 'quote', where 'quote' is copied "
            "verbatim from the computed output below.\n\nComputed output:\n"
            + computed_text
        )
        draft_text = self.transport.draft(prompt)
        return validate_draft(draft_text, computed_text)
