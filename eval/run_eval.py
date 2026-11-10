"""Run all 12 gold scenarios through the full loop and score the decisions.

Usage: python -m eval.run_eval   (from the repo root)

Reports decision accuracy and escalation (NEEDS-HUMAN) precision/recall,
plus a per-scenario table. The numbers printed here are the numbers that
belong in the README results table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analyst_loop.pipeline import run_scenario
from analyst_loop.scenarios import list_scenarios

ESCALATED = "NEEDS-HUMAN"


def main() -> int:
    rows = []
    for spec in list_scenarios():
        result = run_scenario(spec)
        rows.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "expected": spec["expected"],
                "got": result["decision"],
                "correct": result["decision"] == spec["expected"],
            }
        )

    correct = sum(r["correct"] for r in rows)
    accuracy = correct / len(rows)

    tp = sum(1 for r in rows if r["expected"] == ESCALATED and r["got"] == ESCALATED)
    fp = sum(1 for r in rows if r["expected"] != ESCALATED and r["got"] == ESCALATED)
    fn = sum(1 for r in rows if r["expected"] == ESCALATED and r["got"] != ESCALATED)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0

    print("analyst-loop eval — 12 gold scenarios")
    print("=" * 78)
    for r in rows:
        mark = "OK  " if r["correct"] else "MISS"
        print(f"{mark} {r['id']}  {r['name']:<48} expected {r['expected']:<12} got {r['got']}")
    print("=" * 78)
    print(f"decision accuracy:      {accuracy:.3f} ({correct}/{len(rows)})")
    print(f"escalation precision:   {precision:.3f} ({tp}/{tp + fp})")
    print(f"escalation recall:      {recall:.3f} ({tp}/{tp + fn})")
    return 0 if correct == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
