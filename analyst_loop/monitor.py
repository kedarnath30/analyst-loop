"""Change detection on daily product metrics.

A rolling z-score against a seasonal (same-weekday) baseline. The baseline
for a given date is the same weekday over the previous ``baseline_weeks``
weeks, so weekly seasonality does not itself trigger alerts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Alert:
    """One detected metric movement."""

    metric: str
    date: str  # ISO date
    value: float
    baseline_mean: float
    baseline_sd: float
    z: float
    magnitude_pct: float  # (value - baseline) / baseline * 100
    direction: str  # "up" or "down"
    severity: str  # "warning" | "critical"


@dataclass
class DataQualityResult:
    """Result of the missing-days check on a daily series."""

    expected_days: int
    present_days: int
    missing_days: int
    missing_fraction: float
    ok: bool


def check_data_quality(
    df: pd.DataFrame, max_missing_fraction: float = 0.10
) -> DataQualityResult:
    """Count missing calendar days in the daily series.

    ``df`` must have a ``date`` column (datetime). A gap in the calendar
    means the pipeline cannot trust the series.
    """
    dates = pd.to_datetime(df["date"])
    if len(dates) == 0:
        return DataQualityResult(0, 0, 0, 1.0, False)
    full = pd.date_range(dates.min(), dates.max(), freq="D")
    present = len(dates)
    expected = len(full)
    missing = expected - present
    fraction = missing / expected if expected else 1.0
    return DataQualityResult(
        expected_days=expected,
        present_days=present,
        missing_days=missing,
        missing_fraction=fraction,
        ok=fraction <= max_missing_fraction,
    )


def detect_changes(
    df: pd.DataFrame,
    metric: str,
    z_threshold: float = 3.0,
    baseline_weeks: int = 4,
    min_magnitude_pct: float = 2.0,
) -> list[Alert]:
    """Detect movements in ``metric`` against a seasonal baseline.

    The baseline for a date is the same weekday over the previous
    ``baseline_weeks`` weeks. A linear trend fitted to those baseline
    points is extrapolated to today, so steady growth does not itself
    trigger alerts; the residual spread (floored at 1% of the predicted
    value, because a spread estimated from a handful of points is
    unreliable) is the z denominator. Alert when |z| >= ``z_threshold``
    and the relative move is at least ``min_magnitude_pct``. Severity is
    "critical" at |z| >= 5, else "warning".
    """
    series = df.sort_values("date").reset_index(drop=True)
    values = series[metric].to_numpy(dtype=float)
    dates = pd.to_datetime(series["date"])
    alerts: list[Alert] = []

    for i in range(len(series)):
        offsets = np.array(
            [-7 * w for w in range(1, baseline_weeks + 1) if i - 7 * w >= 0],
            dtype=float,
        )
        if len(offsets) < 2:
            continue
        baseline = values[[i - 7 * w for w in range(1, baseline_weeks + 1) if i - 7 * w >= 0]]
        if len(offsets) >= 3:
            slope, intercept = np.polyfit(offsets, baseline, 1)
            predicted = float(intercept)  # value at offset 0 (today)
            residuals = baseline - (slope * offsets + intercept)
            dof = max(len(offsets) - 2, 1)
            sd = float(np.sqrt(np.sum(residuals**2) / dof))
        else:
            predicted = float(np.mean(baseline))
            sd = float(np.std(baseline, ddof=1))
        sd = max(sd, 0.01 * abs(predicted))
        if sd <= 0:
            continue
        x = float(values[i])
        z = (x - predicted) / sd
        magnitude_pct = (x - predicted) / predicted * 100.0 if predicted != 0 else 0.0
        if abs(z) >= z_threshold and abs(magnitude_pct) >= min_magnitude_pct:
            alerts.append(
                Alert(
                    metric=metric,
                    date=str(dates.iloc[i].date()),
                    value=x,
                    baseline_mean=predicted,
                    baseline_sd=sd,
                    z=float(z),
                    magnitude_pct=float(magnitude_pct),
                    direction="down" if z < 0 else "up",
                    severity="critical" if abs(z) >= 5.0 else "warning",
                )
            )
    return alerts
