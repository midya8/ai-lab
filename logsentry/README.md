# logsentry

Dependency-light anomaly detection for DevOps logs. Parses raw log lines, mines message templates (masking IPs, UUIDs, hex, numbers, paths), buckets events into time windows, and flags:

- **volume_spike / volume_drop**: robust z-score (median/MAD) against a rolling baseline
- **error_burst**: windows where the ERROR/FATAL ratio jumps
- **new_template**: log message shapes never seen before

## Try it
```
pip install -r requirements.txt
PYTHONPATH=src python -m logsentry.cli --demo
PYTHONPATH=src python -m logsentry.cli app.log --window 60 --z 4 --json
python -m pytest
```
Demo output finds the injected 8x traffic spike at 12:20 and the DB error burst at 12:25 (with its new template).

## Next
Per-template rate detectors, streaming mode, Prometheus exporter.

## Per-template spikes (v0.2)
`--per-template` tracks each template's count per window and flags `template_spike` when one surges (robust z) even if total volume looks flat.

## Streaming (v0.3)
`StreamDetector` (src/logsentry/stream.py) ingests events one at a time and emits anomalies as soon as each window closes, with bounded memory (`history=`). Output is tested identical to batch `detect`. CLI: `tail -f app.log | PYTHONPATH=src python -m logsentry.cli --follow`, or `--follow --demo`.

## Prometheus exporter (v0.4)
`--follow --metrics-port 9108` serves `/metrics` in Prometheus text format: `logsentry_events_total{level}`, `logsentry_anomalies_total{kind}`, `logsentry_windows_closed_total`, and the latest anomalous window timestamp. Zero dependencies (`metrics.py`).

## Alert webhooks (v0.5)
`--follow --webhook URL [--cooldown 300]` POSTs each anomaly as Slack-compatible JSON (`text` plus structured fields). Identical (kind, detail) alerts are suppressed within the cooldown; delivery failures never crash the stream (`alerts.py`).

## Grafana dashboard (v0.6)
`python src/logsentry/grafana.py dashboard.json` writes an importable Grafana dashboard (events/sec by level, anomalies by kind, window and anomaly stats) over the v0.4 metrics; a test checks every query only references exported metrics.
