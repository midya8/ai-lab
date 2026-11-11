from pathlib import Path

from codesearch.__main__ import main
from codesearch.watch import signature, watch


def _write(p: Path, body: str) -> None:
    p.write_text(body)


def test_watch_resyncs_only_on_change(tmp_path):
    _write(tmp_path / "a.py", "def alpha():\n    return 1\n")
    store = tmp_path / ".codesearch" / "index.json"
    reports = []
    ticks = []

    def sleeper(_):
        ticks.append(1)
        if len(ticks) == 2:  # change a file between poll 1 and poll 2
            _write(tmp_path / "b.py", "def beta():\n    return 2\n")

    n = watch(tmp_path, store, interval=0, on_sync=lambda r, c: reports.append((r, c)),
              max_cycles=4, sleep=sleeper)
    assert n == 2                      # initial sync + exactly one resync
    assert reports[0][0].added == ["a.py"]
    assert reports[1][0].added == ["b.py"] and reports[1][0].reused == 1
    assert reports[1][1] == 2


def test_watch_sees_edit_and_delete(tmp_path):
    a = tmp_path / "a.py"
    _write(a, "def alpha():\n    return 1\n")
    seen = []
    step = []

    def sleeper(_):
        step.append(1)
        if len(step) == 1:
            _write(a, "def alpha():\n    return 1\n\ndef gamma():\n    return 3\n")
        elif len(step) == 2:
            a.unlink()

    watch(tmp_path, tmp_path / "idx.json", interval=0, on_sync=lambda r, c: seen.append((r.changed, r.removed, c)),
          max_cycles=3, sleep=sleeper)
    assert seen[1] == (["a.py"], [], 2)
    assert seen[2][1] == ["a.py"]


def test_signature_ignores_skipped_dirs(tmp_path):
    (tmp_path / "node_modules").mkdir()
    _write(tmp_path / "node_modules" / "x.py", "x = 1\n")
    _write(tmp_path / "ok.py", "y = 1\n")
    assert list(signature(tmp_path)) == ["ok.py"]


def test_cli_watch_bounded(tmp_path, capsys):
    _write(tmp_path / "a.py", "def alpha():\n    return 1\n")
    rc = main(["watch", "--root", str(tmp_path), "--interval", "0", "--cycles", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "1 added" in out
    assert (tmp_path / ".codesearch" / "index.json").exists()
