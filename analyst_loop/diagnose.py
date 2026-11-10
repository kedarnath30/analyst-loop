"""Segment drill-down: attribute a metric movement to slices.

Contribution analysis. For each slice, compare its mean in the event window
against its mean in the baseline window just before the event, and express
the slice's change as a share of total movement:
    contribution = |slice delta| / sum of |slice deltas|.
Shares are computed on absolute deltas so they always sum to 100% — a
slice moving the other way still counts toward the total movement, and
the driver never reads as more than 100% of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class SliceContribution:
    """One slice's share of a metric movement."""

    slice: str
    baseline_mean: float
    event_mean: float
    delta: float
    contribution_pct: float  # delta / total delta * 100


def attribute_movement(
    series_by_slice: dict[str, pd.Series],
    event_start: str,
    event_end: str,
    baseline_days: int = 7,
) -> list[SliceContribution]:
    """Attribute a movement between two windows to slices.

    ``series_by_slice`` maps slice name -> daily series (indexed by date).
    The event window is [event_start, event_end]; the baseline window is
    the ``baseline_days`` days immediately before the event start. Returns
    slices ranked by |contribution|.
    """
    start = pd.Timestamp(event_start)
    end = pd.Timestamp(event_end)
    deltas: dict[str, tuple[float, float, float]] = {}
    for name, series in series_by_slice.items():
        s = series.sort_index()
        baseline = s[(s.index < start) & (s.index >= start - pd.Timedelta(days=baseline_days))]
        event = s[(s.index >= start) & (s.index <= end)]
        if len(baseline) == 0 or len(event) == 0:
            continue
        b_mean = float(baseline.mean())
        e_mean = float(event.mean())
        deltas[name] = (b_mean, e_mean, e_mean - b_mean)

    total_delta = sum(abs(d) for _, _, d in deltas.values())
    out: list[SliceContribution] = []
    for name, (b_mean, e_mean, delta) in deltas.items():
        contrib = (abs(delta) / total_delta * 100.0) if abs(total_delta) > 1e-9 else 0.0
        out.append(
            SliceContribution(
                slice=name,
                baseline_mean=b_mean,
                event_mean=e_mean,
                delta=delta,
                contribution_pct=float(contrib),
            )
        )
    out.sort(key=lambda c: abs(c.contribution_pct), reverse=True)
    return out
