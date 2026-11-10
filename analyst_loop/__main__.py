"""CLI: run one scenario and print the decision memo."""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import run_scenario
from .scenarios import get_scenario, list_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="analyst-loop")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run one scenario and print its memo.")
    run_p.add_argument("--scenario", required=True, help="Scenario id, e.g. 01_real_lift")

    sub.add_parser("list", help="List the gold scenarios.")

    args = parser.parse_args(argv)
    if args.command == "list":
        for spec in list_scenarios():
            print(f"{spec['id']:>18}  {spec['name']:<45} expected: {spec['expected']}")
        return 0

    spec = get_scenario(args.scenario)
    result = run_scenario(spec)
    print(json.dumps(result["memo"], indent=2))
    print()
    print("---")
    print()
    print(result["markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
