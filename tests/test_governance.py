"""Tests for governance.py: every decision path."""

from analyst_loop.data import generate_daily_metrics, generate_experiment
from analyst_loop.experiment import analyze_experiment
from analyst_loop.governance import govern_experiment, govern_monitor
from analyst_loop.monitor import check_data_quality, detect_changes


def _exp_df(**kwargs):
    params = dict(
        seed=11, n_per_arm=20000, base_mean=100.0, base_sd=40.0,
        pre_mean=100.0, pre_sd=40.0, beta=0.3, lift=5.0,
    )
    params.update(kwargs)
    return generate_experiment(**params)


def test_ship_on_real_lift():
    d = govern_experiment(analyze_experiment(_exp_df(seed=11)))
    assert d.label == "SHIP" and "SHP-01" in d.check_ids


def test_hold_on_noise():
    d = govern_experiment(analyze_experiment(_exp_df(seed=12, n_per_arm=10000, lift=0.0)))
    assert d.label == "HOLD" and "EFF-02" in d.check_ids


def test_iterate_on_novelty():
    d = govern_experiment(analyze_experiment(
        _exp_df(seed=13, n_per_arm=10000, lift=0.0,
                novelty={"early_lift": 8.0, "late_lift": 0.0})))
    assert d.label == "ITERATE" and "NOV-01" in d.check_ids


def test_needs_human_on_srm():
    d = govern_experiment(analyze_experiment(
        _exp_df(seed=16, n_per_arm=10000, assign_prob=0.55)))
    assert d.label == "NEEDS-HUMAN" and "SRM-01" in d.check_ids


def test_hold_on_guardrail_breach():
    d = govern_experiment(analyze_experiment(
        _exp_df(seed=15, n_per_arm=10000, guardrail_lift=-4.0)))
    assert d.label == "HOLD" and "GRD-01" in d.check_ids


def test_needs_human_on_underpowered():
    d = govern_experiment(analyze_experiment(
        _exp_df(seed=20, n_per_arm=300, lift=2.0)))
    assert d.label == "NEEDS-HUMAN" and "PWR-01" in d.check_ids


def test_needs_human_on_peeking():
    df = _exp_df(seed=20, n_per_arm=4000, lift=3.0)
    sub = df[df["day"] < 7]
    d = govern_experiment(analyze_experiment(
        sub, mde=5.0, info_fraction=0.5, stopped_at_interim=True))
    assert d.label == "NEEDS-HUMAN" and "SEQ-01" in d.check_ids


def test_monitor_hold_when_quiet():
    df = generate_daily_metrics(seed=1, noise_sd=0.005)
    dq = check_data_quality(df)
    d = govern_monitor(detect_changes(df, "dau"), dq)
    assert d.label == "HOLD" and "MON-02" in d.check_ids


def test_monitor_needs_human_on_data_gap():
    df = generate_daily_metrics(seed=22, missing_days=[48, 49, 50, 51, 52, 53, 54])
    dq = check_data_quality(df)
    d = govern_monitor(detect_changes(df, "dau"), dq)
    assert d.label == "NEEDS-HUMAN" and "DQ-01" in d.check_ids


def test_monitor_iterate_on_single_slice_drop():
    df = generate_daily_metrics(
        seed=19, drop={"platform": "iOS", "start_day": 42, "magnitude": -0.12}
    )
    dq = check_data_quality(df)
    alerts = detect_changes(df, "dau")
    d = govern_monitor(alerts, dq, top_contribution=96.0, top_slice="iOS")
    assert d.label == "ITERATE" and "MON-01" in d.check_ids
