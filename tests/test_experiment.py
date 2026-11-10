"""Tests for experiment.py: estimation and safeguards."""

import numpy as np
from scipy import stats

from analyst_loop.data import generate_experiment
from analyst_loop.experiment import (
    analyze_experiment,
    cuped_analysis,
    obrien_fleming_critical_p,
    power_two_sample,
    sequential_check,
    srm_test,
    welch_ttest,
)


def _lift_df(**kwargs):
    params = dict(
        seed=11, n_per_arm=20000, base_mean=100.0, base_sd=40.0,
        pre_mean=100.0, pre_sd=40.0, beta=0.3, lift=5.0,
    )
    params.update(kwargs)
    return generate_experiment(**params)


def test_welch_matches_scipy():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 40, 500)
    b = rng.normal(105, 40, 500)
    mine = welch_ttest(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert abs(abs(mine.t_stat) - abs(ref.statistic)) < 1e-9
    assert abs(mine.p_value - ref.pvalue) < 1e-9


def test_ci_narrows_with_n():
    small = welch_ttest(np.random.default_rng(0).normal(100, 40, 100),
                        np.random.default_rng(1).normal(105, 40, 100))
    big = welch_ttest(np.random.default_rng(0).normal(100, 40, 10000),
                      np.random.default_rng(1).normal(105, 40, 10000))
    assert (big.ci_high - big.ci_low) < (small.ci_high - small.ci_low)


def test_cuped_reduces_variance_when_correlated():
    df = _lift_df(seed=52, n_per_arm=3000, base_sd=60.0, beta=1.2, lift=3.0)
    r = analyze_experiment(df, mde=5.0)
    assert r.cuped.se < r.raw.se
    assert abs(r.cuped.theta - 1.2) < 0.15


def test_cuped_rescue_pattern():
    df = _lift_df(seed=52, n_per_arm=3000, base_sd=60.0, beta=1.2, lift=3.0)
    r = analyze_experiment(df, mde=5.0)
    assert r.raw.p_value > 0.05  # raw is noisy
    assert r.cuped.p_value < 0.01  # CUPED is significant
    assert r.primary == "cuped"


def test_srm_balanced_passes_and_imbalanced_flags():
    assert not srm_test(10000, 10000).flagged
    imbalanced = srm_test(9000, 11000)
    assert imbalanced.flagged and imbalanced.p_value < 0.001


def test_obrien_fleming_boundaries():
    assert abs(obrien_fleming_critical_p(1.0) - 0.05) < 1e-9
    assert abs(obrien_fleming_critical_p(0.5) - 0.0056) < 0.001


def test_sequential_peek_flagged_only_when_unjustified():
    # Naive-significant at 50% info, below the OBF boundary, stopped early.
    flagged = sequential_check(0.017, info_fraction=0.5, stopped_at_interim=True)
    assert flagged.peeked_flag
    # Same p, but it clears the boundary -> justified stop.
    crossed = sequential_check(0.002, info_fraction=0.5, stopped_at_interim=True)
    assert not crossed.peeked_flag and crossed.crossed_boundary
    # Full-information look is never a peek.
    final = sequential_check(0.017, info_fraction=1.0, stopped_at_interim=False)
    assert not final.peeked_flag


def test_guardrail_breach_detection():
    breach = analyze_experiment(_lift_df(seed=15, n_per_arm=10000, guardrail_lift=-4.0))
    assert breach.guardrail.breached
    clean = analyze_experiment(_lift_df(seed=11))
    assert not clean.guardrail.breached


def test_heterogeneity_harm_flag():
    df = _lift_df(seed=18, n_per_arm=10000, lift=5.0, slice_effects={"IN": -9.0})
    r = analyze_experiment(df)
    assert r.heterogeneity.harm_slices, "expected a harmed slice"
    assert any("country=IN" in s for s in r.heterogeneity.harm_slices)
    assert r.primary_stats.diff > 0  # overall still positive


def test_simpson_detection():
    df = generate_experiment(
        seed=14, n_per_arm=10000, base_mean=100.0, base_sd=40.0,
        pre_mean=100.0, pre_sd=40.0, beta=0.3, lift=-3.0,
        slice_means={"iOS": 150.0, "Android": 80.0},
        slice_mix={"control": {"iOS": 0.2}, "treatment": {"iOS": 0.8}},
    )
    r = analyze_experiment(df)
    assert r.primary_stats.diff > 0  # aggregate says positive
    assert r.heterogeneity.simpson


def test_power_high_vs_low_n():
    assert power_two_sample(10000, 40.0, 5.0) > 0.99
    assert power_two_sample(300, 40.0, 5.0) < 0.50


def test_novelty_flagged_when_effect_fades():
    faded = analyze_experiment(
        _lift_df(seed=13, n_per_arm=10000, lift=0.0,
                 novelty={"early_lift": 8.0, "late_lift": 0.0})
    )
    assert faded.novelty is not None and faded.novelty.flagged
    sustained = analyze_experiment(_lift_df(seed=11))
    assert sustained.novelty is not None and not sustained.novelty.flagged


def test_cuped_theta_matches_beta():
    df = generate_experiment(
        seed=52, n_per_arm=3000, base_mean=100.0, base_sd=60.0,
        pre_mean=100.0, pre_sd=40.0, beta=1.2, lift=3.0,
    )
    c = df[df["arm"] == 0]
    t = df[df["arm"] == 1]
    cuped = cuped_analysis(
        c["pre"].to_numpy(), c["post"].to_numpy(),
        t["pre"].to_numpy(), t["post"].to_numpy(),
    )
    assert abs(cuped.theta - 1.2) < 0.15
