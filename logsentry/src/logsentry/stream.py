"""Streaming detector: feed events one at a time, anomalies emit as each window closes.

Uses the same rules as the batch `detect` (volume spike/drop, error burst, new template)
and is tested to produce identical output. Memory is bounded by `history` windows.
"""
from collections import Counter, deque
from datetime import timedelta
import numpy as np
from .detect import Anomaly, _bucket, robust_z

ERR = ("ERROR", "FATAL", "CRITICAL")


class StreamDetector:
    def __init__(self, window=60, warmup=5, z_thresh=4.0, error_ratio=0.3, history=None):
        self.window, self.warmup = window, warmup
        self.z_thresh, self.error_ratio = z_thresh, error_ratio
        self.volume = deque(maxlen=history)
        self.seen = set()
        self.cur, self.cur_events, self.n_closed = None, [], 0

    def feed(self, event):
        """Add an event (timestamps must be non-decreasing); return anomalies from any windows it closed."""
        w = _bucket(event.ts, self.window)
        out = []
        if self.cur is None:
            self.cur = w
        elif w < self.cur:
            raise ValueError("out-of-order event: stream must be time-ordered")
        while self.cur < w:
            out.extend(self._close())
            self.cur += timedelta(seconds=self.window)
        self.cur_events.append(event)
        return out

    def flush(self):
        """Close the in-progress window (call at end of input)."""
        if self.cur is None or not self.cur_events:
            return []
        out = self._close()
        self.cur = None
        return out

    def _close(self):
        evs, w, out = self.cur_events, self.cur, []
        if self.n_closed >= self.warmup:
            z = robust_z(len(evs), self.volume)
            if abs(z) >= self.z_thresh:
                kind = "volume_spike" if z > 0 else "volume_drop"
                out.append(Anomaly(kind, w, round(z, 2), f"{len(evs)} events (baseline median {np.median(self.volume):.0f})"))
            if evs:
                errs = [e for e in evs if e.level in ERR]
                if len(errs) / len(evs) >= self.error_ratio and len(errs) >= 5:
                    top = Counter(e.template for e in errs).most_common(1)[0][0]
                    out.append(Anomaly("error_burst", w, round(len(errs) / len(evs), 2), f"{len(errs)}/{len(evs)} errors, top: {top}"))
            for tpl in sorted({e.template for e in evs} - self.seen):
                out.append(Anomaly("new_template", w, 1.0, tpl))
        self.seen.update(e.template for e in evs)
        self.volume.append(len(evs))
        self.n_closed += 1
        self.cur_events = []
        return sorted(out, key=lambda a: (a.window, a.kind))
