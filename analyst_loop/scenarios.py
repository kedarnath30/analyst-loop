"""Scenario loading: 12 gold scenarios with known ground truth."""

from __future__ import annotations

import json
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "data" / "scenarios"


def list_scenarios() -> list[dict]:
    """All scenario specs, sorted by id."""
    specs = []
    for path in sorted(SCENARIO_DIR.glob("*.json")):
        specs.append(json.loads(path.read_text()))
    return specs


def get_scenario(name: str) -> dict:
    """Fetch one scenario by id (e.g. '01_real_lift') or by name."""
    for spec in list_scenarios():
        if spec["id"] == name or spec["name"] == name:
            return spec
    raise KeyError(f"Unknown scenario: {name!r}")
