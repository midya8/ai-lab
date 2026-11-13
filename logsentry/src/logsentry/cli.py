import argparse, json, sys
from .parser import parse_line
from .detect import detect
from .synth import generate


def main(argv=None):
    p = argparse.ArgumentParser(prog="logsentry", description="Detect anomalies in log streams")
    p.add_argument("file", nargs="?", help="log file (default: stdin)")
    p.add_argument("--demo", action="store_true", help="analyze a synthetic log with injected incidents")
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--z", type=float, default=4.0)
    p.add_argument("--json", action="store_true")
    p.add_argument("--per-template", action="store_true", help="also flag spikes in individual log templates")
    p.add_argument("--incidents", action="store_true", help="batch mode: group anomalies into incidents (one line per outage)")
    p.add_argument("--gap", type=int, default=2, help="with --incidents: max windows between anomalies of one incident")
    p.add_argument("--follow", action="store_true", help="streaming mode: read stdin/file line by line, print anomalies as windows close")
    p.add_argument("--metrics-port", type=int, help="with --follow: serve Prometheus metrics at http://127.0.0.1:PORT/metrics")
    p.add_argument("--webhook", help="with --follow: POST each anomaly (Slack-compatible JSON) to this URL")
    p.add_argument("--cooldown", type=int, default=300, help="seconds before an identical alert re-fires (default 300)")
    a = p.parse_args(argv)
    if a.follow:
        from .metrics import MetricsDetector, serve
        sd = MetricsDetector(window=a.window, z_thresh=a.z)
        if a.metrics_port:
            serve(sd, a.metrics_port)
        from .alerts import WebhookNotifier
        notifier = WebhookNotifier(a.webhook, cooldown=a.cooldown) if a.webhook else None
        src = generate() if a.demo else (open(a.file) if a.file else sys.stdin)
        def show(xs):
            if notifier:
                notifier.notify(xs)
            for x in xs:
                print(f"{x.window:%H:%M} {x.kind:13} score={x.score:<7} {x.detail}", flush=True)
        for line in src:
            e = parse_line(line.rstrip("\n"))
            if e:
                show(sd.feed(e))
        show(sd.flush())
        return 0
    if a.demo:
        lines = generate()
    else:
        lines = (open(a.file) if a.file else sys.stdin).read().splitlines()
    events = [e for e in map(parse_line, lines) if e]
    found = detect(events, window=a.window, z_thresh=a.z, per_template=a.per_template)
    if a.incidents:
        from .incidents import group_incidents
        inc = group_incidents(found, window=a.window, gap=a.gap)
        if a.json:
            print(json.dumps([{"start": i.start.isoformat(), "end": i.end.isoformat(), "severity": i.severity,
                               "kinds": i.kinds, "peak": i.peak, "anomalies": len(i.anomalies)} for i in inc], indent=2))
        else:
            print(f"{len(events)} events, {len(found)} anomalies, {len(inc)} incidents")
            for i in inc:
                print(i.summary())
        return 0
    if a.json:
        print(json.dumps([{**x.__dict__, "window": x.window.isoformat()} for x in found], indent=2))
    else:
        print(f"{len(events)} events, {len(found)} anomalies")
        for x in found:
            print(f"{x.window:%H:%M} {x.kind:13} score={x.score:<7} {x.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
