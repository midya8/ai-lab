"""Webhook alerting with per-(kind, detail) cooldown and Slack-compatible JSON payloads."""
import json
import urllib.request


class WebhookNotifier:
    def __init__(self, url, cooldown=300, min_score=0.0, kinds=None, sender=None):
        self.url, self.cooldown, self.min_score = url, cooldown, min_score
        self.kinds = set(kinds) if kinds else None
        self._last = {}
        self._send = sender or self._post
        self.sent = 0
        self.suppressed = 0

    @staticmethod
    def payload(a):
        text = f"[logsentry] {a.kind} at {a.window:%Y-%m-%d %H:%M} score={a.score}: {a.detail}"
        return {"text": text, "kind": a.kind, "window": a.window.isoformat(), "score": a.score, "detail": a.detail}

    def _post(self, body):
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()

    def notify(self, anomalies):
        """Send each qualifying anomaly unless the same (kind, detail) fired within cooldown seconds
        of its window time. Returns the anomalies actually sent. Delivery errors never raise."""
        out = []
        for a in anomalies:
            if self.kinds and a.kind not in self.kinds or a.score < self.min_score:
                continue
            key, ts = (a.kind, a.detail), a.window.timestamp()
            if key in self._last and ts - self._last[key] < self.cooldown:
                self.suppressed += 1
                continue
            self._last[key] = ts
            try:
                self._send(self.payload(a))
                self.sent += 1
                out.append(a)
            except Exception:
                self._last.pop(key, None)
        return out
