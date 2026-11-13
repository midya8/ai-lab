"""Group raw anomalies into incidents: anomalies whose windows are within `gap` windows
of each other are merged, so one outage becomes one page instead of dozens of alerts."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class Incident:
    start: datetime
    end: datetime
    kinds: dict = field(default_factory=dict)  # kind -> count
    peak: float = 0.0
    anomalies: list = field(default_factory=list)

    @property
    def severity(self) -> str:
        n = len(self.kinds)
        if n >= 3 or self.peak >= 20:
            return "high"
        return "medium" if n == 2 or self.peak >= 8 else "low"

    def summary(self) -> str:
        ks = ", ".join(f"{k} x{c}" for k, c in sorted(self.kinds.items()))
        return f"[{self.severity}] {self.start:%H:%M:%S}-{self.end:%H:%M:%S} {ks} (peak score {self.peak:g})"


def group_incidents(anomalies, window=60, gap=2):
    """Merge anomalies whose windows are <= gap*window seconds apart (chained).
    Peak is the max |score|; note error_burst scores are ratios (<=1) so volume/template
    z-scores dominate it."""
    out, cur = [], None
    for a in sorted(anomalies, key=lambda a: a.window):
        if cur is not None and a.window - cur.end <= timedelta(seconds=gap * window):
            cur.end = max(cur.end, a.window)
        else:
            cur = Incident(a.window, a.window)
            out.append(cur)
        cur.kinds[a.kind] = cur.kinds.get(a.kind, 0) + 1
        cur.peak = max(cur.peak, abs(a.score))
        cur.anomalies.append(a)
    return out
