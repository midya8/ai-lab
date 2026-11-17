# logsentry

Anomaly detection for application logs. It parses raw lines, mines message templates (masking IPs, UUIDs, numbers, paths), buckets events into time windows, and flags volume spikes and drops (robust z-score against a rolling median/MAD baseline), error bursts, per-template spikes, and message shapes it has never seen before.

```bash
cd logsentry
pip install -r requirements.txt
PYTHONPATH=src python -m logsentry.cli --demo
PYTHONPATH=src python -m logsentry.cli app.log --window 60 --z 4 --json
tail -f app.log | PYTHONPATH=src python -m logsentry.cli --follow --metrics-port 9108
python -m pytest -q
```

The demo finds the injected 8x traffic spike at 12:20 and the DB error burst at 12:25, along with its new template.

Streaming mode emits anomalies as each window closes and gives the same output as batch mode. On top of that there's a Prometheus `/metrics` endpoint, a Grafana dashboard generator (`python src/logsentry/grafana.py dashboard.json`), Slack-style webhooks with a cooldown, and `--incidents` to group a burst of anomalies into one incident.

There's also a seasonal detector that compares each window to the same phase in previous cycles. On a synthetic daily cycle the plain detector raises false alarms and the seasonal one doesn't, while still catching a 3x surge at the peak.

Only tested on synthetic logs. The incident severity is a hand-set heuristic.
