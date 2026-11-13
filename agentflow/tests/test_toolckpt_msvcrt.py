import threading

from agentflow import ToolJournal
import agentflow.toolckpt as tc


class FakeMsvcrt:
    LK_NBLCK, LK_UNLCK = 2, 0

    def __init__(self, fail_first=0):
        self.fail, self.calls = fail_first, []

    def locking(self, fd, mode, n):
        if mode == self.LK_NBLCK and self.fail > 0:
            self.fail -= 1
            raise OSError("locked")
        self.calls.append(mode)


def test_msvcrt_fallback_locks_and_unlocks(tmp_path, monkeypatch):
    fake = FakeMsvcrt(fail_first=3)
    monkeypatch.setattr(tc, "fcntl", None)
    monkeypatch.setattr(tc, "msvcrt", fake)
    j = ToolJournal(str(tmp_path / "j.jsonl"))
    j.store("f", 0, "t", {}, "o")
    assert fake.calls == [fake.LK_NBLCK, fake.LK_UNLCK]
    assert j.lookup("f", 0, "t", {})["obs"] == "o"


def test_msvcrt_lock_released_on_error(tmp_path, monkeypatch):
    fake = FakeMsvcrt()
    monkeypatch.setattr(tc, "fcntl", None)
    monkeypatch.setattr(tc, "msvcrt", fake)
    j = ToolJournal(str(tmp_path / "j.jsonl"))
    try:
        with j._flock():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert fake.calls[-1] == fake.LK_UNLCK


def test_no_locking_module_still_works(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "fcntl", None)
    monkeypatch.setattr(tc, "msvcrt", None)
    j = ToolJournal(str(tmp_path / "j.jsonl"))
    j.store("f", 0, "t", {}, "o")
    assert len(j) == 1
