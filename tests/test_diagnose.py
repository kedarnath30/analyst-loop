"""Tests for diagnose.py: segment attribution."""

from analyst_loop.data import generate_daily_metrics
from analyst_loop.diagnose import attribute_movement


def _series(df):
    return {
        "iOS": df.set_index("date")["dau_ios"],
        "Android": df.set_index("date")["dau_android"],
    }


def test_attributes_drop_to_ios():
    df = generate_daily_metrics(
        seed=19, drop={"platform": "iOS", "start_day": 42, "magnitude": -0.12}
    )
    contribs = attribute_movement(_series(df), "2026-09-14", "2026-09-14")
    assert contribs[0].slice == "iOS"
    assert contribs[0].contribution_pct > 80.0


def test_contributions_sum_to_100():
    df = generate_daily_metrics(
        seed=19, drop={"platform": "iOS", "start_day": 42, "magnitude": -0.12}
    )
    contribs = attribute_movement(_series(df), "2026-09-14", "2026-09-14")
    total = sum(c.contribution_pct for c in contribs)
    assert abs(total - 100.0) < 1.0


def test_no_movement_means_no_dominant_slice():
    df = generate_daily_metrics(seed=1, noise_sd=0.001)
    contribs = attribute_movement(_series(df), "2026-08-20", "2026-08-20")
    # Deltas are per-platform noise (~0.8% of each platform's DAU); no
    # slice dominates the movement the way a real drop would.
    assert all(abs(c.delta) < 1500 for c in contribs)
    assert max(abs(c.contribution_pct) for c in contribs) < 80.0
