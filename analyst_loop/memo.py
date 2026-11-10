"""Decision-memo generator.

Every claim in the memo cites the metric, the date range, and the check ID
that produced it — the same provenance discipline as a redline citation.
A claim without a citation is not written.
"""

from __future__ import annotations

from .governance import Decision


def build_memo(
    scenario_id: str,
    scenario_name: str,
    decision: Decision,
    claims: list[dict],
    numbers: dict,
) -> dict:
    """Assemble the structured memo.

    ``claims`` is a list of {"text": ..., "metric": ..., "date_range": ...,
    "check_id": ...}. ``numbers`` holds the key computed figures.
    """
    return {
        "scenario_id": scenario_id,
        "scenario": scenario_name,
        "decision": decision.label,
        "summary": [c["text"] for c in claims],
        "claims": claims,
        "numbers": numbers,
        "decision_reasons": decision.reasons,
        "check_ids": decision.check_ids,
    }


def render_markdown(memo: dict) -> str:
    """Render the structured memo as Markdown."""
    lines = [
        f"# Decision memo — {memo['scenario']}",
        "",
        f"**Decision: {memo['decision']}**",
        "",
        "## What happened",
    ]
    for claim in memo["claims"]:
        lines.append(
            f"- {claim['text']} "
            f"[{claim['metric']} · {claim['date_range']} · {claim['check_id']}]"
        )
    lines.append("")
    lines.append("## Key numbers")
    for key, value in memo["numbers"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## Why this decision")
    for reason in memo["decision_reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(f"Checks: {', '.join(memo['check_ids'])}")
    return "\n".join(lines) + "\n"
