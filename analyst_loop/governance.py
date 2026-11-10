"""The responsible-AI governance gate.

One decision per analysis, with the check IDs that produced it. The gate
escalates to a human whenever the evidence is ambiguous, the data is
suspect, or any slice of users is harmed — a wrong ship costs more than a
slow ship.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .experiment import ExperimentResult
from .monitor import Alert, DataQualityResult

DECISIONS = ("SHIP", "HOLD", "ITERATE", "NEEDS-HUMAN")


@dataclass
class Decision:
    """A governed decision, with its provenance."""

    label: str
    reasons: list[str] = field(default_factory=list)
    check_ids: list[str] = field(default_factory=list)


# Practical-significance floor: an effect smaller than this is trivial even
# when statistically significant. 1.0 on a base outcome of 100 = 1% relative.
PRACTICAL_THRESHOLD = 1.0


def govern_experiment(result: ExperimentResult) -> Decision:
    """Decide SHIP / HOLD / ITERATE / NEEDS-HUMAN for an experiment."""
    p = result.primary_stats

    if result.missing_pre_fraction > 0.05:
        return Decision(
            "NEEDS-HUMAN",
            ["Pre-period covariate missing for "
             f"{result.missing_pre_fraction:.1%} of users; CUPED and balance "
             "cannot be trusted."],
            ["DQ-01"],
        )
    if result.srm.flagged:
        return Decision(
            "NEEDS-HUMAN",
            [f"Sample-ratio mismatch: {result.n_control:,} control vs "
             f"{result.n_treatment:,} treatment "
             f"(chi2={result.srm.chi2:.1f}, p={result.srm.p_value:.2g}); "
             "assignment is broken, so the comparison is not clean."],
            ["SRM-01"],
        )
    if result.sequential.peeked_flag:
        look = result.sequential.looks[0]
        return Decision(
            "NEEDS-HUMAN",
            [f"Test stopped early at {look['info_fraction']:.0%} of planned "
             f"information with a naively significant p={look['p_value']:.3f}, "
             f"but the O'Brien-Fleming boundary at that look is "
             f"p<{look['critical_p']:.4f}. Do not ship on a peeked result."],
            ["SEQ-01"],
        )
    if result.guardrail.breached:
        return Decision(
            "HOLD",
            [f"Guardrail metric moved {result.guardrail.diff:+.2f} "
             f"(95% CI {result.guardrail.ci_low:+.2f} to "
             f"{result.guardrail.ci_high:+.2f}, p={result.guardrail.p_value:.2g}); "
             "the primary gain is not worth the guardrail loss."],
            ["GRD-01"],
        )
    if result.heterogeneity.simpson:
        return Decision(
            "NEEDS-HUMAN",
            ["Overall effect and every significant slice disagree in sign — "
             "a Simpson's-paradox pattern. The aggregate number is an "
             "artifact of slice mix, not a real effect."],
            ["HET-02"],
        )
    if result.heterogeneity.harm_slices:
        harmed = ", ".join(result.heterogeneity.harm_slices)
        return Decision(
            "NEEDS-HUMAN",
            [f"Overall effect is positive, but these slices are significantly "
             f"harmed: {harmed}. A ship would trade one segment's loss for "
             "another's gain."],
            ["HET-01"],
        )
    if result.power < 0.80 and p.ci_low < result.mde and p.ci_high > -result.mde:
        return Decision(
            "NEEDS-HUMAN",
            [f"Power is only {result.power:.0%} for the minimum detectable "
             f"effect of {result.mde:.1f}; the 95% CI "
             f"({p.ci_low:+.2f} to {p.ci_high:+.2f}) is too wide to rule a "
             "meaningful effect in or out. Collect more data."],
            ["PWR-01"],
        )
    if result.novelty is not None and result.novelty.flagged:
        return Decision(
            "ITERATE",
            [f"Novelty effect: early-period lift {result.novelty.early_diff:+.2f} "
             f"(p={result.novelty.early_p:.2g}) faded to "
             f"{result.novelty.late_diff:+.2f} (p={result.novelty.late_p:.2g}) "
             "late. The effect does not survive users getting used to it."],
            ["NOV-01"],
        )
    if p.p_value >= 0.05:
        if p.ci_low > -result.mde and p.ci_high < result.mde:
            return Decision(
                "HOLD",
                [f"No significant effect (p={p.p_value:.2f}) and the 95% CI "
                 f"({p.ci_low:+.2f} to {p.ci_high:+.2f}) sits inside the "
                 f"±{result.mde:.1f} minimum-effect band: conclusively no "
                 "meaningful effect."],
                ["EFF-02"],
            )
        return Decision(
            "NEEDS-HUMAN",
            [f"Effect is not significant (p={p.p_value:.2f}) and the CI is "
             "too wide to call it null. A human should judge whether to "
             "extend the test."],
            ["EFF-01"],
        )
    if p.ci_low < PRACTICAL_THRESHOLD:
        return Decision(
            "NEEDS-HUMAN",
            [f"Effect is significant (p={p.p_value:.2g}) but the 95% CI "
             f"({p.ci_low:+.2f} to {p.ci_high:+.2f}) reaches below the "
             f"practical-significance floor of {PRACTICAL_THRESHOLD:.1f}; the "
             "real gain may be trivial."],
            ["EFF-01"],
        )
    return Decision(
        "SHIP",
        [f"Primary effect {p.diff:+.2f} (95% CI {p.ci_low:+.2f} to "
         f"{p.ci_high:+.2f}, p={p.p_value:.2g}) via {result.primary}; no "
         "guardrail breach, no slice harm, no assignment or data issue."],
        ["SHP-01"],
    )


def govern_monitor(
    alerts: list[Alert],
    dq: DataQualityResult,
    top_contribution: float | None = None,
    top_slice: str | None = None,
) -> Decision:
    """Decide for a monitoring scenario (no experiment running)."""
    if not dq.ok:
        return Decision(
            "NEEDS-HUMAN",
            [f"Data-quality gap: {dq.missing_days} of {dq.expected_days} days "
             f"missing ({dq.missing_fraction:.0%}). Do not trust any alert "
             "until the feed is repaired."],
            ["DQ-01"],
        )
    if not alerts:
        return Decision(
            "HOLD",
            ["No metric movement beyond the alert threshold. Nothing to act on."],
            ["MON-02"],
        )
    worst = max(alerts, key=lambda a: abs(a.z))
    if top_contribution is not None and top_contribution >= 80.0 and top_slice:
        return Decision(
            "ITERATE",
            [f"{worst.metric} moved {worst.magnitude_pct:+.1f}% on {worst.date} "
             f"(z={worst.z:.1f}, {worst.severity}); {top_slice} accounts for "
             f"{top_contribution:.0f}% of the movement. Investigate the "
             "slice-specific cause before acting on the total."],
            ["MON-01", "DIA-01"],
        )
    return Decision(
        "ITERATE",
        [f"{worst.metric} moved {worst.magnitude_pct:+.1f}% on {worst.date} "
         f"(z={worst.z:.1f}, {worst.severity}); no single slice explains it. "
         "Diagnose further before acting."],
        ["MON-01"],
    )
