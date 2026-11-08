"""python -m agentflow log RUNLOG.jsonl [--compact]"""
import argparse
import sys

from .runlog import RunLog


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="agentflow")
    sub = ap.add_subparsers(dest="cmd", required=True)
    lg = sub.add_parser("log", help="summarize a run log")
    lg.add_argument("path")
    lg.add_argument("--compact", action="store_true", help="drop superseded and replayed records")
    a = ap.parse_args(argv)
    log = RunLog(a.path)
    if a.compact:
        print(f"dropped {log.compact()} line(s)")
    print(log.format_runs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
