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
