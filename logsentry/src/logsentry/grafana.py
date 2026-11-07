"""Generate a Grafana dashboard (JSON) for the logsentry Prometheus metrics."""
import json
import sys

def _panel(pid, title, expr, x, y, w=12, h=8, ptype="timeseries", legend="{{label}}"):
    return {"id": pid, "title": title, "type": ptype,
            "datasource": {"type": "prometheus", "uid": "${DS_PROMETHEUS}"},
            "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "targets": [{"expr": expr, "legendFormat": legend, "refId": "A"}]}

def build_dashboard(title="logsentry"):
    panels = [
        _panel(1, "Events/sec by level", "sum by (level) (rate(logsentry_events_total[1m]))", 0, 0, legend="{{level}}"),
        _panel(2, "Anomalies/min by kind", "sum by (kind) (increase(logsentry_anomalies_total[5m]))/5", 12, 0, legend="{{kind}}"),
        _panel(3, "Windows closed (total)", "logsentry_windows_closed_total", 0, 8, 6, 4, "stat", "windows"),
        _panel(4, "Total anomalies", "sum(logsentry_anomalies_total)", 6, 8, 6, 4, "stat", "anomalies"),
        _panel(5, "Last anomalous window", "logsentry_last_anomaly_window_timestamp_seconds*1000", 12, 8, 12, 4, "stat", "last"),
    ]
    panels[4]["fieldConfig"] = {"defaults": {"unit": "dateTimeFromNow"}}
    return {"title": title, "uid": "logsentry", "schemaVersion": 39, "version": 1,
            "refresh": "10s", "time": {"from": "now-1h", "to": "now"}, "tags": ["logsentry"],
            "templating": {"list": [{"name": "DS_PROMETHEUS", "type": "datasource", "query": "prometheus"}]},
            "panels": panels}

def main(argv=None):
    out = json.dumps(build_dashboard(), indent=2)
    if argv and argv[0] != "-":
        open(argv[0], "w").write(out + "\n")
    else:
        print(out)

if __name__ == "__main__":
    main(sys.argv[1:])
