"""Detectors: robust rate spikes (median/MAD), unseen templates, error bursts."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class Anomaly:
    kind: str
    window: datetime
    score: float
    detail: str


def _bucket(ts: datetime, size: int) -> datetime:
    s = int(ts.timestamp()) // size * size
    return datetime.fromtimestamp(s)


def robust_z(x: float, history) -> float:
    h = np.asarray(history, float)
    med = np.median(h)
    mad = np.median(np.abs(h - med)) * 1.4826
    mad = max(mad, 1.0)  # floor avoids infinite scores on flat series
    return float((x - med) / mad)


def detect(events, window=60, warmup=5, z_thresh=4.0, error_ratio=0.3, per_template=False, min_count=10):
    """Return anomalies sorted by time. The first `warmup` windows train the baseline."""
    events = sorted(events, key=lambda e: e.ts)
    if not events:
        return []
    by_win = defaultdict(list)
    for e in events:
        by_win[_bucket(e.ts, window)].append(e)
    start, end = min(by_win), max(by_win)
    wins, t = [], start
    while t <= end:
        wins.append(t)
        t += timedelta(seconds=window)

    out, seen, volume = [], set(), []
    tpl_hist = defaultdict(list)
    for i, w in enumerate(wins):
        evs = by_win.get(w, [])
        training = i < warmup
        if not training:
            z = robust_z(len(evs), volume)
            if abs(z) >= z_thresh:
                kind = "volume_spike" if z > 0 else "volume_drop"
                out.append(Anomaly(kind, w, round(z, 2), f"{len(evs)} events (baseline median {np.median(volume):.0f})"))
            if evs:
                errs = sum(e.level in ("ERROR", "FATAL", "CRITICAL") for e in evs)
                if errs / len(evs) >= error_ratio and errs >= 5:
                    top = Counter(e.template for e in evs if e.level in ("ERROR", "FATAL", "CRITICAL")).most_common(1)[0][0]
                    out.append(Anomaly("error_burst", w, round(errs / len(evs), 2), f"{errs}/{len(evs)} errors, top: {top}"))
            for tpl in sorted({e.template for e in evs} - seen):
                out.append(Anomaly("new_template", w, 1.0, tpl))
        if per_template:
            cnt = Counter(e.template for e in evs)
            if not training:
                for tpl, hist in tpl_hist.items():
                    if len(hist) < i:  # pad windows before first sighting
                        hist[:0] = [0] * (i - len(hist))
                    c = cnt.get(tpl, 0)
                    z = robust_z(c, hist)
                    if z >= z_thresh and c >= min_count:
                        out.append(Anomaly("template_spike", w, round(z, 2), f"{tpl}: {c} in window (median {np.median(hist):.0f})"))
            for tpl in set(tpl_hist) | set(cnt):
                h = tpl_hist[tpl]
                h.extend([0] * (i - len(h)))
                h.append(cnt.get(tpl, 0))
        seen.update(e.template for e in evs)
        volume.append(len(evs))
    return sorted(out, key=lambda a: (a.window, a.kind))
