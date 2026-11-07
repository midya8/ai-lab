"""Prometheus text-format metrics for logsentry (no client library needed)."""
from collections import Counter
from .stream import StreamDetector


class MetricsDetector:
    """Wraps StreamDetector and tracks counters renderable in Prometheus exposition format."""

    def __init__(self, **kw):
        self.sd = StreamDetector(**kw)
        self.events = Counter()      # by level
        self.anomalies = Counter()   # by kind
        self.last_window_ts = None

    def feed(self, event):
        self.events[event.level] += 1
        return self._track(self.sd.feed(event))

    def flush(self):
        return self._track(self.sd.flush())

    def _track(self, found):
        for a in found:
            self.anomalies[a.kind] += 1
            self.last_window_ts = a.window.timestamp()
        return found

    def render(self):
        L = ["# HELP logsentry_events_total Log events processed, by level.",
             "# TYPE logsentry_events_total counter"]
        L += [f'logsentry_events_total{{level="{k}"}} {v}' for k, v in sorted(self.events.items())]
        L += ["# HELP logsentry_anomalies_total Anomalies detected, by kind.",
              "# TYPE logsentry_anomalies_total counter"]
        L += [f'logsentry_anomalies_total{{kind="{k}"}} {v}' for k, v in sorted(self.anomalies.items())]
        L += ["# HELP logsentry_windows_closed_total Analysis windows closed.",
              "# TYPE logsentry_windows_closed_total counter",
              f"logsentry_windows_closed_total {self.sd.n_closed}"]
        if self.last_window_ts is not None:
            L += ["# HELP logsentry_last_anomaly_window_timestamp_seconds Start of the latest anomalous window.",
                  "# TYPE logsentry_last_anomaly_window_timestamp_seconds gauge",
                  f"logsentry_last_anomaly_window_timestamp_seconds {self.last_window_ts:.0f}"]
        return "\n".join(L) + "\n"


def serve(md, port=9108):
    """Serve md.render() at /metrics from a background thread; returns the server."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_response(404); self.end_headers(); return
            body = md.render().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
