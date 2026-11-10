"""Integration tests: all 12 gold scenarios + the CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from analyst_loop.pipeline import run_scenario
from analyst_loop.scenarios import get_scenario, list_scenarios

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("spec", list_scenarios(), ids=lambda s: s["id"])
def test_gold_scenario_decision(spec):
    result = run_scenario(spec)
    assert result["decision"] == spec["expected"], (
        f"{spec['id']}: expected {spec['expected']}, got {result['decision']}"
    )


def test_scenario_ids_unique_and_complete():
    specs = list_scenarios()
    assert len(specs) == 12
    assert len({s["id"] for s in specs}) == 12


def test_cli_run_prints_json_and_markdown():
    proc = subprocess.run(
        [sys.executable, "-m", "analyst_loop", "run", "--scenario", "01_real_lift"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    json_part = proc.stdout.split("\n---\n")[0]
    memo = json.loads(json_part)
    assert memo["decision"] == "SHIP"
    assert "Decision memo" in proc.stdout


def test_cli_list_shows_twelve_scenarios():
    proc = subprocess.run(
        [sys.executable, "-m", "analyst_loop", "list"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=60,
    )
    assert proc.returncode == 0
    assert proc.stdout.count("expected:") == 12


def test_cli_unknown_scenario_fails():
    proc = subprocess.run(
        [sys.executable, "-m", "analyst_loop", "run", "--scenario", "nope"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=60,
    )
    assert proc.returncode != 0
