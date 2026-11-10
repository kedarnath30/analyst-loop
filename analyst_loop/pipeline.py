"""Pipeline: run one scenario through the full loop.

Monitor -> Diagnose -> Experiment -> Decide, with a memo at the end.
Every step is deterministic given the scenario seed.
"""

from __future__ import annotations

import pandas as pd

from .data import ANCHOR_DATE, generate_daily_metrics, generate_experiment
from .diagnose import attribute_movement
from .experiment import analyze_experiment
from .governance import govern_experiment, govern_monitor
from .memo import build_memo, render_markdown
from .monitor import check_data_quality, detect_changes

POST_START = ANCHOR_DATE
POST_END = "2026-08-16"  # 14-day post window from the anchor


def _experiment_claims(result) -> list[dict]:
    p = result.primary_stats
    claims = [
        {
            "text": (
                f"Primary effect {p.diff:+.2f} (95% CI {p.ci_low:+.2f} to "
                f"{p.ci_high:+.2f}, p={p.p_value:.3g}) using {result.primary} "
                f"(n={result.n_control:,} control, {result.n_treatment:,} treatment)."
            ),
            "metric": "post outcome",
            "date_range": f"{POST_START} to {POST_END}",
            "check_id": "EXP-01",
        },
    ]
    if result.primary == "cuped":
        claims.append(
            {
                "text": (
                    f"CUPED theta {result.cuped.theta:.2f} cut the standard error "
                    f"from {result.raw.se:.2f} to {result.cuped.se:.2f}."
                ),
                "metric": "post outcome",
                "date_range": f"{POST_START} to {POST_END}",
                "check_id": "CUP-01",
            }
        )
    claims.append(
        {
            "text": (
                f"Guardrail moved {result.guardrail.diff:+.2f} "
                f"(p={result.guardrail.p_value:.3g})."
                if result.guardrail.breached
                else "Guardrail metric did not move significantly."
            ),
            "metric": "guardrail outcome",
            "date_range": f"{POST_START} to {POST_END}",
            "check_id": "GRD-01",
        }
    )
    if result.heterogeneity.simpson:
        claims.append(
            {
                "text": "Overall effect and every significant slice disagree "
                "in sign (Simpson's paradox).",
                "metric": "post outcome by slice",
                "date_range": f"{POST_START} to {POST_END}",
                "check_id": "HET-02",
            }
        )
    elif result.heterogeneity.harm_slices:
        claims.append(
            {
                "text": "Slices significantly harmed: "
                + ", ".join(result.heterogeneity.harm_slices)
                + ".",
                "metric": "post outcome by slice",
                "date_range": f"{POST_START} to {POST_END}",
                "check_id": "HET-01",
            }
        )
    else:
        claims.append(
            {
                "text": "No slice is significantly harmed.",
                "metric": "post outcome by slice",
                "date_range": f"{POST_START} to {POST_END}",
                "check_id": "HET-01",
            }
        )
    claims.append(
        {
            "text": (
                f"Sample-ratio mismatch detected "
                f"(chi2={result.srm.chi2:.1f}, p={result.srm.p_value:.2g})."
                if result.srm.flagged
                else "Assignment is balanced; no sample-ratio mismatch."
            ),
            "metric": "assignment counts",
            "date_range": f"{POST_START} to {POST_END}",
            "check_id": "SRM-01",
        }
    )
    if result.novelty is not None and result.novelty.flagged:
        claims.append(
            {
                "text": (
                    f"Early-period lift {result.novelty.early_diff:+.2f} faded to "
                    f"{result.novelty.late_diff:+.2f} late — a novelty effect."
                ),
                "metric": "post outcome by period",
                "date_range": f"{POST_START} to {POST_END}",
                "check_id": "NOV-01",
            }
        )
    return claims


def _run_experiment(spec: dict) -> dict:
    params = dict(spec["experiment"])
    seed = int(spec["seed"])
    df = generate_experiment(seed=seed, **params)
    stopped = bool(spec.get("stopped_at_interim", False))
    info_fraction = float(spec.get("info_fraction", 1.0))
    if stopped:
        # The team stopped the test early: judge only the interim snapshot
        # (users enrolled in the first half of the 14-day window).
        df = df[df["day"] < 7].copy()
    result = analyze_experiment(
        df,
        mde=float(spec.get("mde", 5.0)),
        info_fraction=info_fraction,
        stopped_at_interim=stopped,
    )
    decision = govern_experiment(result)
    p = result.primary_stats
    numbers = {
        "primary effect": f"{p.diff:+.2f} (95% CI {p.ci_low:+.2f} to {p.ci_high:+.2f}, p={p.p_value:.3g})",
        "primary method": result.primary,
        "CUPED theta": f"{result.cuped.theta:.2f}",
        "power for MDE": f"{result.power:.0%}",
        "guardrail effect": f"{result.guardrail.diff:+.2f} (p={result.guardrail.p_value:.3g})",
        "SRM chi2": f"{result.srm.chi2:.1f} (p={result.srm.p_value:.2g})",
    }
    memo = build_memo(
        spec["id"], spec["name"], decision, _experiment_claims(result), numbers
    )
    return {
        "scenario_id": spec["id"],
        "name": spec["name"],
        "kind": "experiment",
        "expected": spec["expected"],
        "decision": decision.label,
        "memo": memo,
        "markdown": render_markdown(memo),
    }


def _run_monitor(spec: dict) -> dict:
    params = dict(spec["daily"])
    seed = int(spec["seed"])
    metric = spec["metric"]
    df = generate_daily_metrics(seed=seed, **params)
    dq = check_data_quality(df)
    alerts = detect_changes(df, metric)
    contributions = []
    top_slice = None
    top_contribution = None
    if alerts:
        worst = max(alerts, key=lambda a: abs(a.z))
        series_by_slice = {
            "iOS": df.set_index("date")["dau_ios"],
            "Android": df.set_index("date")["dau_android"],
        }
        # Attribute over a 7-day window ending at the worst alert: a
        # single day is too noisy to attribute (per-platform daily noise
        # is ~0.8%), while a week of the sustained drop attributes cleanly.
        event_end = pd.Timestamp(worst.date)
        event_start = event_end - pd.Timedelta(days=6)
        contributions = attribute_movement(
            series_by_slice,
            str(event_start.date()),
            str(event_end.date()),
            baseline_days=7,
        )
        if contributions:
            top_slice = contributions[0].slice
            top_contribution = contributions[0].contribution_pct
    decision = govern_monitor(alerts, dq, top_contribution, top_slice)
    date_range = f"{df['date'].min().date()} to {df['date'].max().date()}"
    claims: list[dict] = []
    if not dq.ok:
        claims.append(
            {
                "text": f"Data-quality gap: {dq.missing_days} of "
                f"{dq.expected_days} days missing "
                f"({dq.missing_fraction:.0%}).",
                "metric": metric,
                "date_range": date_range,
                "check_id": "DQ-01",
            }
        )
    for alert in alerts:
        claims.append(
            {
                "text": f"{metric} moved {alert.magnitude_pct:+.1f}% on "
                f"{alert.date} (z={alert.z:.1f}, {alert.severity}).",
                "metric": metric,
                "date_range": date_range,
                "check_id": "MON-01",
            }
        )
    if contributions:
        claims.append(
            {
                "text": f"{contributions[0].slice} accounts for "
                f"{contributions[0].contribution_pct:.0f}% of the movement.",
                "metric": metric,
                "date_range": date_range,
                "check_id": "DIA-01",
            }
        )
    if not claims:
        claims.append(
            {
                "text": "No metric movement beyond the alert threshold.",
                "metric": metric,
                "date_range": date_range,
                "check_id": "MON-02",
            }
        )
    numbers = {
        "alerts": str(len(alerts)),
        "missing days": f"{dq.missing_days} of {dq.expected_days}",
        "top slice contribution": (
            f"{top_slice}: {top_contribution:.0f}%"
            if top_slice is not None
            else "n/a"
        ),
    }
    memo = build_memo(spec["id"], spec["name"], decision, claims, numbers)
    return {
        "scenario_id": spec["id"],
        "name": spec["name"],
        "kind": "monitor",
        "expected": spec["expected"],
        "decision": decision.label,
        "memo": memo,
        "markdown": render_markdown(memo),
    }


def run_scenario(spec: dict) -> dict:
    """Run one scenario spec through the full loop."""
    if spec["kind"] == "experiment":
        return _run_experiment(spec)
    if spec["kind"] == "monitor":
        return _run_monitor(spec)
    raise ValueError(f"Unknown scenario kind: {spec['kind']!r}")
