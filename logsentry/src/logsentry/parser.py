"""Log parsing and template mining (a lightweight Drain-style masker)."""
import re
from dataclasses import dataclass
from datetime import datetime

LINE = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\d[T ]\d\d:\d\d:\d\d)\S*\s+(?P<level>[A-Z]+)\s+(?P<msg>.*)$")
MASKS = [
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b"), "<IP>"),
    (re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"), "<UUID>"),
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
    (re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?:ms|s|kb|mb|gb|%)?\b", re.I), "<NUM>"),
    (re.compile(r"/[\w./-]+"), "<PATH>"),
]


@dataclass(frozen=True)
class LogEvent:
    ts: datetime
    level: str
    template: str
    raw: str


def mine_template(msg: str) -> str:
    for rx, tok in MASKS:
        msg = rx.sub(tok, msg)
    return re.sub(r"\s+", " ", msg).strip()


def parse_line(line: str):
    m = LINE.match(line.strip())
    if not m:
        return None
    ts = datetime.fromisoformat(m["ts"].replace(" ", "T"))
    return LogEvent(ts, m["level"], mine_template(m["msg"]), line.rstrip("\n"))
