from agentflow import ToolJournal


def test_auto_compacts_when_over_max_lines(tmp_path):
    p = tmp_path / "j.jsonl"
    j = ToolJournal(str(p), max_lines=5)
    for i in range(20):
        j.store("fp", 0, "t", {}, f"v{i}")  # same key rewritten
    assert j.auto_compactions >= 3
    assert len(p.read_text().splitlines()) <= 5
    assert j.lookup("fp", 0, "t", {})["obs"] == "v19"


def test_counts_existing_lines_on_open(tmp_path):
    p = tmp_path / "j.jsonl"
    j = ToolJournal(str(p))
    for i in range(6):
        j.store("fp", 0, "t", {}, str(i))
    j2 = ToolJournal(str(p), max_lines=6)
    j2.store("fp", 1, "t", {}, "x")
    assert j2.auto_compactions == 1
    assert len(p.read_text().splitlines()) == 2
    assert len(ToolJournal(str(p))) == 2


def test_no_auto_compaction_by_default(tmp_path):
    j = ToolJournal(str(tmp_path / "j.jsonl"))
    for i in range(50):
        j.store("fp", 0, "t", {}, str(i))
    assert j.auto_compactions == 0
