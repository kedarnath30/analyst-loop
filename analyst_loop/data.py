"""Synthetic data generation.

Everything produced here is fake data for a fictional social app, generated
with a seeded RNG so every run is deterministic. The anchor date is fixed
(2026-08-03, a Monday) so memos can cite stable date ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANCHOR_DATE = "2026-08-03"

# Day-of-week multipliers (Mon=0 .. Sun=6): weekends run hotter.
DOW_FACTORS = np.array([0.97, 1.00, 1.01, 1.00, 0.99, 1.06, 1.07])


def generate_daily_metrics(
    seed: int,
    days: int = 56,
    base_dau: float = 100_000.0,
    trend_per_day: float = 0.001,
    noise_sd: float = 0.015,
    drop: dict | None = None,
    missing_days: list[int] | None = None,
) -> pd.DataFrame:
    """Daily product metrics for a fictional social app.

    Parameters
    ----------
    seed: RNG seed (deterministic output).
    days: number of days to generate.
    base_dau: DAU on day 0 before trend/seasonality.
    trend_per_day: linear growth rate.
    noise_sd: multiplicative lognormal noise on DAU.
    drop: optional injected event, e.g.
        {"platform": "iOS", "start_day": 42, "magnitude": -0.12}
        (platform may also be "all"). Magnitude is a relative change.
    missing_days: day indices (0-based) whose rows are dropped entirely,
        simulating a data-quality gap.

    Returns a DataFrame with columns: date, dau, dau_ios, dau_android,
    retention_d7, watch_min_per_user, conversion.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(ANCHOR_DATE, periods=days, freq="D")
    t = np.arange(days)
    dow = dates.day_of_week.to_numpy()

    dau = (
        base_dau
        * (1.0 + trend_per_day * t)
        * DOW_FACTORS[dow]
        * np.exp(rng.normal(0.0, noise_sd, days))
    )

    ios_share = 0.55
    dau_ios = dau * ios_share * np.exp(rng.normal(0.0, 0.008, days))
    dau_android = dau * (1.0 - ios_share) * np.exp(rng.normal(0.0, 0.008, days))

    if drop is not None:
        start = int(drop["start_day"])
        mag = float(drop["magnitude"])
        platform = drop.get("platform", "all")
        if platform in ("iOS", "all"):
            dau_ios[start:] *= 1.0 + mag
        if platform in ("Android", "all"):
            dau_android[start:] *= 1.0 + mag
        dau = dau_ios + dau_android

    df = pd.DataFrame(
        {
            "date": dates,
            "dau": dau,
            "dau_ios": dau_ios,
            "dau_android": dau_android,
            "retention_d7": 0.42 + rng.normal(0.0, 0.010, days),
            "watch_min_per_user": 38.0 + rng.normal(0.0, 1.5, days),
            "conversion": 0.032 + rng.normal(0.0, 0.0015, days),
        }
    )

    if missing_days:
        df = df.drop(index=[i for i in missing_days if 0 <= i < len(df)])
        df = df.reset_index(drop=True)
    return df


def generate_experiment(
    seed: int,
    n_per_arm: int = 10_000,
    base_mean: float = 100.0,
    base_sd: float = 40.0,
    pre_mean: float = 100.0,
    pre_sd: float = 40.0,
    beta: float = 0.3,
    lift: float = 0.0,
    novelty: dict | None = None,
    slice_effects: dict | None = None,
    slice_means: dict | None = None,
    slice_mix: dict | None = None,
    guardrail_lift: float = 0.0,
    guardrail_base: float = 50.0,
    guardrail_sd: float = 15.0,
    assign_prob: float = 0.5,
) -> pd.DataFrame:
    """User-level experiment data for a fictional social app.

    Every user has a pre-period covariate (``pre``), a post-period primary
    outcome (``post``), a guardrail outcome, an enrolment day, a period
    (early/late half of the post window), and segment fields
    (platform, country, cohort).

    Parameters
    ----------
    seed: RNG seed.
    n_per_arm: nominal users per arm (total N = 2 * n_per_arm, split by
        ``assign_prob``).
    base_mean, base_sd: control post-outcome mean/sd (overridden per slice
        by ``slice_means``).
    pre_mean, pre_sd: pre-period covariate distribution.
    beta: regression coefficient of post on (pre - pre_mean); drives the
        pre/post correlation that CUPED exploits.
    lift: additive treatment effect on the post outcome.
    novelty: {"early_lift": x, "late_lift": y} — replaces ``lift`` by period.
    slice_effects: {"<segment value>": delta} — extra treatment effect for
        users in that slice (matched against platform, country, or cohort).
    slice_means: {"iOS": 150, "Android": 80} — per-platform base means.
    slice_mix: {"control": {"iOS": 0.2}, "treatment": {"iOS": 0.8}} —
        per-arm platform proportions (Simpson's paradox construction).
    guardrail_lift: treatment effect on the guardrail outcome.
    assign_prob: P(user is assigned to treatment).
    """
    rng = np.random.default_rng(seed)
    n_total = 2 * n_per_arm
    arm = (rng.random(n_total) < assign_prob).astype(int)  # 1 = treatment

    # Segments. Platform mix can differ by arm (Simpson construction).
    platform = np.empty(n_total, dtype=object)
    for a, key in ((0, "control"), (1, "treatment")):
        mask = arm == a
        p_ios = 0.55
        if slice_mix is not None:
            p_ios = float(slice_mix.get(key, {}).get("iOS", 0.55))
        platform[mask] = rng.choice(
            ["iOS", "Android"], size=int(mask.sum()), p=[p_ios, 1.0 - p_ios]
        )
    country = rng.choice(["US", "IN", "GB"], size=n_total, p=[0.60, 0.25, 0.15])
    cohort = rng.choice(["new", "returning"], size=n_total, p=[0.30, 0.70])
    day = rng.integers(0, 14, size=n_total)
    period = np.where(day < 7, "early", "late")

    pre = rng.normal(pre_mean, pre_sd, n_total)
    noise_sd = float(np.sqrt(max(base_sd**2 - (beta * pre_sd) ** 2, 1.0)))
    post_base = base_mean + beta * (pre - pre_mean) + rng.normal(0.0, noise_sd, n_total)
    if slice_means is not None:
        for value, mean in slice_means.items():
            mask = platform == value
            post_base[mask] = (
                float(mean)
                + beta * (pre[mask] - pre_mean)
                + rng.normal(0.0, noise_sd, int(mask.sum()))
            )

    effect = np.full(n_total, float(lift))
    if novelty is not None:
        effect = np.where(
            period == "early",
            float(novelty.get("early_lift", lift)),
            float(novelty.get("late_lift", lift)),
        )
    if slice_effects is not None:
        for value, delta in slice_effects.items():
            mask = (platform == value) | (country == value) | (cohort == value)
            effect[mask] += float(delta)

    post = post_base + np.where(arm == 1, effect, 0.0)
    guardrail = rng.normal(guardrail_base, guardrail_sd, n_total) + np.where(
        arm == 1, float(guardrail_lift), 0.0
    )

    return pd.DataFrame(
        {
            "user_id": np.arange(n_total),
            "arm": arm,
            "day": day,
            "period": period,
            "platform": platform,
            "country": country,
            "cohort": cohort,
            "pre": pre,
            "post": post,
            "guardrail": guardrail,
        }
    )
