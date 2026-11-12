"""Seasonality-aware volume detection.

The plain detector compares each window to a rolling median of *all* recent
windows, so a normal daily peak looks like a spike. Here each window is
compared only to windows at the same phase of the cycle (e.g. same minute of
the previous days), so regular peaks and troughs are expected.
"""
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from .detect import Anomaly, _bucket


def window_counts(events, window=60):
    """Dense list of (window_start, count) from first to last event."""
    by = defaultdict(int)
    for e in events:
        by[_bucket(e.ts, window)] += 1
    if not by:
        return []
    t, end, out = min(by), max(by), []
    while t <= end:
        out.append((t, by.get(t, 0)))
        t += timedelta(seconds=window)
    return out


def seasonal_z(x, hist, floor=1.0):
    """Robust z with a Poisson noise floor: scale >= sqrt(median), since counts
    from few same-phase samples give an unreliably small MAD."""
    h = np.asarray(hist, float)
    med = np.median(h)
    mad = np.median(np.abs(h - med)) * 1.4826
    return float((x - med) / max(mad, math.sqrt(max(med, 0.0)), floor))


def detect_seasonal(events, window=60, period=24, min_cycles=2, z_thresh=4.0, floor=1.0):
    """Flag volume anomalies versus same-phase history.

    period: windows per cycle. A window is scored only once `min_cycles`
    earlier cycles exist at its phase. Uses median/MAD of those samples with
    a scale floored at sqrt(median) (Poisson noise) and `floor`.
    """
    if period < 2 or min_cycles < 2:
        raise ValueError("period>=2 and min_cycles>=2 required")
    series = window_counts(events, window)
    out = []
    for i, (w, c) in enumerate(series):
        hist = [series[j][1] for j in range(i - period, -1, -period)]
        if len(hist) < min_cycles:
            continue
        z = seasonal_z(c, hist, floor)
        if abs(z) >= z_thresh:
            kind = "seasonal_spike" if z > 0 else "seasonal_drop"
            out.append(Anomaly(kind, w, round(z, 2),
                               f"{c} events (same-phase median {np.median(hist):.0f}, {len(hist)} cycles)"))
    return out


def generate_seasonal(cycles=5, period=24, base=60, amp=0.8, seed=0, incident=None):
    """Synthetic log lines whose rate follows a sinusoid (one window = 1 minute).

    incident: optional (window_index, multiplier) injecting a surge.
    """
    from .parser import parse_line  # noqa: F401  (format reference)
    rng = random.Random(seed)
    t0 = datetime(2025, 1, 1, 0, 0, 0)
    lines = []
    for m in range(cycles * period):
        rate = base * (1 + amp * math.sin(2 * math.pi * (m % period) / period))
        n = max(1, int(rate + rng.gauss(0, rate ** 0.5)))
        if incident and m == incident[0]:
            n = int(n * incident[1])
        for _ in range(n):
            ts = t0 + timedelta(minutes=m, seconds=rng.random() * 59)
            lines.append((ts, f"INFO GET /api/items/{rng.randint(1, 999)} 200 {rng.randint(3, 90)}ms"))
    lines.sort()
    return [f"{ts.isoformat(timespec='seconds')} {msg}" for ts, msg in lines]
