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
    a = p.parse_args(argv)
    if a.demo:
        lines = generate()
    else:
        lines = (open(a.file) if a.file else sys.stdin).read().splitlines()
    events = [e for e in map(parse_line, lines) if e]
    found = detect(events, window=a.window, z_thresh=a.z, per_template=a.per_template)
    if a.json:
        print(json.dumps([{**x.__dict__, "window": x.window.isoformat()} for x in found], indent=2))
    else:
        print(f"{len(events)} events, {len(found)} anomalies")
        for x in found:
            print(f"{x.window:%H:%M} {x.kind:13} score={x.score:<7} {x.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
