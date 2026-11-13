import json

from agentflow import ToolJournal


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def test_ttl_expires_entries(tmp_path):
    c = Clock()
    j = ToolJournal(str(tmp_path / "j.jsonl"), ttl_s=60, clock=c)
    j.store("fp", 0, "send", {"to": "a"}, "sent")
    assert j.lookup("fp", 0, "send", {"to": "a"})["obs"] == "sent"
    c.t += 61
    assert j.lookup("fp", 0, "send", {"to": "a"}) is None


def test_legacy_entries_without_ts_never_expire(tmp_path):
    p = tmp_path / "j.jsonl"
    p.write_text(json.dumps({"fp": "f", "seq": 0, "tool": "t", "args": {}, "obs": "x"}) + "\n")
    c = Clock()
    j = ToolJournal(str(p), ttl_s=1, clock=c)
    c.t += 10**6
    assert j.lookup("f", 0, "t", {}) is not None


def test_compact_dedupes_expired_and_dead_fps(tmp_path):
    p = str(tmp_path / "j.jsonl")
    c = Clock()
    j = ToolJournal(p, ttl_s=100, clock=c)
    j.store("old", 0, "t", {}, "o")
    c.t += 200
    j.store("a", 0, "t", {}, "v1")
    j.store("a", 0, "t", {}, "v2")  # duplicate key, newest wins
    j.store("b", 0, "t", {}, "w")
    removed = j.compact(live_fps={"a"})
    assert removed == 3 and len(j) == 1
    j2 = ToolJournal(p)
    assert j2.lookup("a", 0, "t", {})["obs"] == "v2"
    assert j2.lookup("b", 0, "t", {}) is None
    assert len(open(p).read().splitlines()) == 1


def test_compact_missing_file(tmp_path):
    assert ToolJournal(str(tmp_path / "none.jsonl")).compact() == 0
