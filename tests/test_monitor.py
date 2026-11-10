"""Tests for monitor.py: change detection and data quality."""

import pandas as pd

from analyst_loop.data import generate_daily_metrics
from analyst_loop.monitor import check_data_quality, detect_changes


def test_detects_injected_drop():
    df = generate_daily_metrics(
        seed=19, drop={"platform": "iOS", "start_day": 42, "magnitude": -0.12}
    )
    alerts = detect_changes(df, "dau")
    # Scope to the drop window (day 42 = 2026-09-14): the drop must alert.
    in_window = [a for a in alerts if a.date >= "2026-09-14"]
    assert in_window, "expected an alert inside the drop window"
    worst = max(in_window, key=lambda a: abs(a.z))
    assert worst.direction == "down"
    assert worst.z < -3.0
    assert worst.magnitude_pct < -2.0


def test_no_alert_on_quiet_series():
    df = generate_daily_metrics(seed=1, noise_sd=0.005)
    assert detect_changes(df, "dau") == []


def test_weekly_seasonality_does_not_alert():
    # Weekends run ~6-7% hotter by construction; the same-weekday baseline
    # must absorb that so no alert fires.
    df = generate_daily_metrics(seed=2, noise_sd=0.005)
    alerts = detect_changes(df, "dau")
    assert all(a.date != "2026-08-08" for a in alerts)  # a Saturday
    assert len(alerts) == 0


def test_severity_scales_with_z():
    df = generate_daily_metrics(
        seed=19, drop={"platform": "all", "start_day": 42, "magnitude": -0.20}
    )
    alerts = detect_changes(df, "dau")
    assert any(a.severity == "critical" for a in alerts)


def test_data_quality_clean():
    df = generate_daily_metrics(seed=1)
    dq = check_data_quality(df)
    assert dq.ok and dq.missing_days == 0 and dq.missing_fraction == 0.0


def test_data_quality_gap():
    df = generate_daily_metrics(seed=22, missing_days=[48, 49, 50, 51, 52, 53, 54])
    dq = check_data_quality(df)
    assert not dq.ok
    assert dq.missing_days == 7
    assert abs(dq.missing_fraction - 7 / 56) < 1e-9
