"""A/B experiment analysis: estimation, variance reduction, and safeguards.

Covers Welch's t-test with confidence intervals, CUPED variance reduction,
sample-ratio-mismatch (SRM) detection, O'Brien-Fleming sequential monitoring,
guardrail checks, heterogeneity/slice checks (harm + Simpson's paradox),
power analysis, and novelty-effect detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StatsResult:
    """A two-sample difference estimate."""

    diff: float
    se: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_control: int
    n_treatment: int


@dataclass
class CUPEDResult(StatsResult):
    """CUPED-adjusted estimate, with the regression coefficient."""

    theta: float = 0.0


@dataclass
class SRMResult:
    chi2: float
    p_value: float
    flagged: bool


@dataclass
class SequentialResult:
    looks: list[dict]
    naive_significant: bool
    crossed_boundary: bool
    peeked_flag: bool


@dataclass
class GuardrailResult:
    diff: float
    se: float
    p_value: float
    ci_low: float
    ci_high: float
    breached: bool


@dataclass
class SliceResult:
    slice: str
    dimension: str
    diff: float
    se: float
    p_value: float
    n_control: int
    n_treatment: int


@dataclass
class HeterogeneityResult:
    harm_slices: list[str]
    simpson: bool


@dataclass
class NoveltyResult:
    flagged: bool
    early_diff: float
    early_p: float
    late_diff: float
    late_p: float


@dataclass
class ExperimentResult:
    """Full analysis of one experiment."""

    n_control: int
    n_treatment: int
    raw: StatsResult
    cuped: CUPEDResult
    primary: str  # "raw" or "cuped"
    primary_stats: StatsResult
    srm: SRMResult
    sequential: SequentialResult
    guardrail: GuardrailResult
    slices: list[SliceResult] = field(default_factory=list)
    heterogeneity: HeterogeneityResult = field(
        default_factory=lambda: HeterogeneityResult([], False)
    )
    power: float = 0.0
    mde: float = 5.0
    novelty: NoveltyResult | None = None
    missing_pre_fraction: float = 0.0


def welch_ttest(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> StatsResult:
    """Welch's two-sample t-test: b (treatment) minus a (control)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    n_a, n_b = len(a), len(b)
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    var_a, var_b = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = float(np.sqrt(var_a / n_a + var_b / n_b))
    diff = mean_b - mean_a
    if se <= 0:
        return StatsResult(diff, 0.0, 0.0, 1.0, diff, diff, n_a, n_b)
    t_stat = diff / se
    # Welch-Satterthwaite degrees of freedom.
    num = (var_a / n_a + var_b / n_b) ** 2
    den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    dof = num / den if den > 0 else min(n_a, n_b) - 1
    p_value = float(2.0 * stats.t.sf(abs(t_stat), dof))
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, dof))
    return StatsResult(
        diff=diff,
        se=se,
        t_stat=float(t_stat),
        p_value=p_value,
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        n_control=n_a,
        n_treatment=n_b,
    )


def cuped_analysis(
    pre_c: np.ndarray,
    post_c: np.ndarray,
    pre_t: np.ndarray,
    post_t: np.ndarray,
    alpha: float = 0.05,
) -> CUPEDResult:
    """CUPED variance reduction using the pre-period covariate.

    theta is estimated on the pooled sample as cov(post, pre) / var(pre);
    adjusted outcomes are post - theta * (pre - grand mean of pre).
    """
    # Pooled pre/post for the theta estimate.
    pre_pool = np.concatenate([np.asarray(pre_c, dtype=float), np.asarray(pre_t, dtype=float)])
    post_pool = np.concatenate([np.asarray(post_c, dtype=float), np.asarray(post_t, dtype=float)])
    pre_pool = pre_pool[~np.isnan(pre_pool)]
    post_pool = post_pool[~np.isnan(post_pool)]
    var_pre = float(np.var(pre_pool, ddof=1))
    theta = float(np.cov(post_pool, pre_pool, ddof=1)[0, 1] / var_pre) if var_pre > 0 else 0.0
    mean_pre = float(np.mean(pre_pool))
    adj_c = np.asarray(post_c, dtype=float) - theta * (np.asarray(pre_c, dtype=float) - mean_pre)
    adj_t = np.asarray(post_t, dtype=float) - theta * (np.asarray(pre_t, dtype=float) - mean_pre)
    base = welch_ttest(adj_c, adj_t, alpha=alpha)
    return CUPEDResult(
        diff=base.diff,
        se=base.se,
        t_stat=base.t_stat,
        p_value=base.p_value,
        ci_low=base.ci_low,
        ci_high=base.ci_high,
        n_control=base.n_control,
        n_treatment=base.n_treatment,
        theta=theta,
    )


def srm_test(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> SRMResult:
    """Chi-square sample-ratio-mismatch test on arm counts."""
    total = n_control + n_treatment
    exp_c = total * (1.0 - expected_ratio)
    exp_t = total * expected_ratio
    chi2 = (n_control - exp_c) ** 2 / exp_c + (n_treatment - exp_t) ** 2 / exp_t
    p_value = float(stats.chi2.sf(chi2, df=1))
    return SRMResult(chi2=float(chi2), p_value=p_value, flagged=p_value < 0.001)


def obrien_fleming_critical_p(info_fraction: float, alpha: float = 0.05) -> float:
    """O'Brien-Fleming-type critical p-value at an information fraction.

    Lan-DeMets O'Brien-Fleming spending function:
    p_crit(t) = 2 * (1 - Phi(z_{alpha/2} / sqrt(t))).
    At t = 0.5 this is ~0.0056; at t = 1.0 it is 0.05.
    """
    t = min(max(info_fraction, 1e-6), 1.0)
    z = float(stats.norm.ppf(1.0 - alpha / 2.0)) / np.sqrt(t)
    return float(2.0 * stats.norm.sf(z))


def sequential_check(
    primary_p: float, info_fraction: float = 1.0, stopped_at_interim: bool = False
) -> SequentialResult:
    """Check whether an early stop was justified by the sequential boundary.

    Flags a "peek": the result is naively significant (p < 0.05) at an
    interim look but does not cross the O'Brien-Fleming boundary, and the
    team stopped the test there anyway.
    """
    crit = obrien_fleming_critical_p(info_fraction)
    naive = primary_p < 0.05
    crossed = primary_p < crit
    peeked = bool(stopped_at_interim and naive and not crossed)
    return SequentialResult(
        looks=[{"info_fraction": info_fraction, "p_value": primary_p, "critical_p": crit}],
        naive_significant=naive,
        crossed_boundary=crossed,
        peeked_flag=peeked,
    )


def power_two_sample(n_per_arm: int, sd: float, mde: float, alpha: float = 0.05) -> float:
    """Two-sample power (normal approximation) for a given MDE."""
    se = sd * np.sqrt(2.0 / n_per_arm)
    ncp = mde / se
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    return float(stats.norm.cdf(ncp - z) + stats.norm.cdf(-ncp - z))


def analyze_experiment(
    df: pd.DataFrame,
    mde: float = 5.0,
    guardrail_breach: float = 1.0,
    info_fraction: float = 1.0,
    stopped_at_interim: bool = False,
    segment_cols: tuple[str, ...] = ("platform", "country", "cohort"),
) -> ExperimentResult:
    """Run the full analysis on user-level experiment data.

    ``df`` needs columns: arm (0 control / 1 treatment), pre, post,
    guardrail, period (early/late), and the segment columns.
    """
    control = df[df["arm"] == 0]
    treatment = df[df["arm"] == 1]
    missing_pre = float(df["pre"].isna().mean()) if len(df) else 0.0

    raw = welch_ttest(control["post"].to_numpy(), treatment["post"].to_numpy())
    cuped = cuped_analysis(
        control["pre"].to_numpy(),
        control["post"].to_numpy(),
        treatment["pre"].to_numpy(),
        treatment["post"].to_numpy(),
    )
    primary = "cuped" if cuped.se < raw.se else "raw"
    primary_stats = cuped if primary == "cuped" else raw

    srm = srm_test(len(control), len(treatment))
    sequential = sequential_check(primary_stats.p_value, info_fraction, stopped_at_interim)

    guard = welch_ttest(
        control["guardrail"].to_numpy(), treatment["guardrail"].to_numpy()
    )
    guardrail = GuardrailResult(
        diff=guard.diff,
        se=guard.se,
        p_value=guard.p_value,
        ci_low=guard.ci_low,
        ci_high=guard.ci_high,
        breached=bool(guard.p_value < 0.05 and guard.diff < -abs(guardrail_breach)),
    )

    slices: list[SliceResult] = []
    for col in segment_cols:
        for value in sorted(df[col].dropna().unique()):
            mask_c = (control[col] == value)
            mask_t = (treatment[col] == value)
            if int(mask_c.sum()) < 30 or int(mask_t.sum()) < 30:
                continue
            r = welch_ttest(
                control.loc[mask_c, "post"].to_numpy(),
                treatment.loc[mask_t, "post"].to_numpy(),
            )
            slices.append(
                SliceResult(
                    slice=f"{col}={value}",
                    dimension=col,
                    diff=r.diff,
                    se=r.se,
                    p_value=r.p_value,
                    n_control=r.n_control,
                    n_treatment=r.n_treatment,
                )
            )

    overall = primary_stats.diff
    harm = [
        s.slice
        for s in slices
        if s.p_value < 0.05 and ((overall > 0 and s.diff < 0) or (overall < 0 and s.diff > 0))
    ]
    # Simpson's paradox is a per-dimension pattern: within one dimension,
    # every significant slice disagrees in sign with the aggregate, because
    # the aggregate is driven by slice mix. (Other dimensions still carry
    # the mix effect, so they agree with the aggregate — that is expected.)
    simpson = False
    for col in segment_cols:
        dim_slices = [s for s in slices if s.dimension == col and s.p_value < 0.05]
        if (
            len(dim_slices) >= 2
            and all(
                (overall > 0 and s.diff < 0) or (overall < 0 and s.diff > 0)
                for s in dim_slices
            )
        ):
            simpson = True
    heterogeneity = HeterogeneityResult(harm_slices=harm, simpson=simpson)

    pooled_sd = float(
        np.sqrt(
            (np.var(control["post"].to_numpy(), ddof=1) + np.var(treatment["post"].to_numpy(), ddof=1)) / 2.0
        )
    )
    power = power_two_sample(
        min(len(control), len(treatment)), pooled_sd, mde
    )

    novelty = None
    if "period" in df.columns:
        early_c = control[control["period"] == "early"]["post"].to_numpy()
        early_t = treatment[treatment["period"] == "early"]["post"].to_numpy()
        late_c = control[control["period"] == "late"]["post"].to_numpy()
        late_t = treatment[treatment["period"] == "late"]["post"].to_numpy()
        if len(early_c) > 30 and len(late_c) > 30:
            e = welch_ttest(early_c, early_t)
            l = welch_ttest(late_c, late_t)
            flagged = bool(
                e.p_value < 0.05 and e.diff > 0 and l.p_value >= 0.05 and l.diff < 0.25 * e.diff
            )
            novelty = NoveltyResult(
                flagged=flagged,
                early_diff=e.diff,
                early_p=e.p_value,
                late_diff=l.diff,
                late_p=l.p_value,
            )

    return ExperimentResult(
        n_control=len(control),
        n_treatment=len(treatment),
        raw=raw,
        cuped=cuped,
        primary=primary,
        primary_stats=primary_stats,
        srm=srm,
        sequential=sequential,
        guardrail=guardrail,
        slices=slices,
        heterogeneity=heterogeneity,
        power=power,
        mde=mde,
        novelty=novelty,
        missing_pre_fraction=missing_pre,
    )
