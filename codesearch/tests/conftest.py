import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

FILES = {
    "app/retry.py": '''"""Retry helpers."""
import time


def retry_with_backoff(fn, attempts=3, base_delay=0.5):
    """Call fn again after failures, doubling the wait each time (exponential backoff)."""
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            time.sleep(base_delay * 2 ** i)  # wait before trying again
    raise RuntimeError("gave up")


class CircuitBreaker:
    """Stops calling a flaky service after repeated errors."""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.failures = 0

    def allow(self):
        return self.failures < self.threshold

    def record_failure(self):
        self.failures += 1
''',
    "app/textutil.py": '''import re


def parseHTTPHeaders(raw):
    """Split a raw HTTP header block into a dict."""
    out = {}
    for line in raw.splitlines():
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip()
    return out


def slugify(title):
    """Turn a title into a url slug: lowercase words joined by hyphens."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def word_count(text):
    return len(text.split())
''',
    "app/storage.py": '''import json


def save_json_atomically(path, data):
    """Write data to a temp file then rename it over path, so readers never see half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh)
    import os
    os.replace(tmp, path)


def load_json(path):
    with open(path) as fh:
        return json.load(fh)
''',
    "tests/test_retry.py": '''from app.retry import retry_with_backoff


def test_retry_with_backoff_returns_value():
    assert retry_with_backoff(lambda: 1) == 1
''',
}


@pytest.fixture
def repo(tmp_path):
    for rel, text in FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path
