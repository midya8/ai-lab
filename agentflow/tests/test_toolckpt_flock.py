import multiprocessing as mp

from agentflow import ToolJournal


def _writer(path, wid, n):
    j = ToolJournal(path)
    for i in range(n):
        j.store(f"fp{wid}", i, "t", {"i": i}, "x" * 200)


def test_concurrent_processes_no_lost_or_torn_lines(tmp_path):
    p = str(tmp_path / "j.jsonl")
    ps = [mp.Process(target=_writer, args=(p, w, 60)) for w in range(4)]
    [x.start() for x in ps]
    [x.join() for x in ps]
    j = ToolJournal(p)
    assert len(j) == 240


def test_refresh_sees_other_process_entries(tmp_path):
    p = str(tmp_path / "j.jsonl")
    a, b = ToolJournal(p), ToolJournal(p)
    b.store("f", 0, "t", {}, "o")
    assert a.lookup("f", 0, "t", {}) is None
    a.refresh()
    assert a.lookup("f", 0, "t", {})["obs"] == "o"


def test_compact_keeps_entries_from_other_process(tmp_path):
    p = str(tmp_path / "j.jsonl")
    a, b = ToolJournal(p), ToolJournal(p)
    a.store("f", 0, "t", {}, "a")
    b.store("g", 0, "t", {}, "b")
    a.compact()
    assert ToolJournal(p).lookup("g", 0, "t", {}) is not None


def test_autocompact_merges_other_process(tmp_path):
    p = str(tmp_path / "j.jsonl")
    a, b = ToolJournal(p, max_lines=3), ToolJournal(p)
    b.store("other", 0, "t", {}, "b")
    for i in range(4):
        a.store("f", i, "t", {}, "a")
    assert ToolJournal(p).lookup("other", 0, "t", {}) is not None
