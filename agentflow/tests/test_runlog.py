import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agentflow import *
from agentflow.__main__ import main


class Counting:
    def __init__(self, fail_times=0):
        self.calls, self.fail_times = 0, fail_times

    def complete(self, m):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("flaky")
        return f"out{self.calls}"


def build(fail_times=0):
    llms = {n: Counting(fail_times if n == "b" else 0) for n in "abc"}
    ags = [Agent(n, llms[n], ToolRegistry(), system=n) for n in "abc"]
    return Orchestrator(ags), llms


def dag(prompt_c="C {x}{y}"):
    return [Task("x", "a", "X"), Task("y", "b", "Y"), Task("z", "c", prompt_c, deps=["x", "y"])]


def test_fingerprint_chains_through_deps():
    f1 = fingerprints(dag())
    f2 = fingerprints([Task("x", "a", "X changed"), *dag()[1:]])
    assert f1["y"] == f2["y"]            # unrelated branch unchanged
    assert f1["x"] != f2["x"] and f1["z"] != f2["z"]  # edit invalidates downstream
    assert fingerprints(dag(prompt_c="other"))["z"] != f1["z"]


def test_log_records_every_task(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, _ = build()
    o.run(dag(), log=log)
    es = log.entries()
    assert sorted(e["task"] for e in es) == ["x", "y", "z"]
    assert all(e["status"] == "ok" and not e["resumed"] for e in es)
    assert len({e["run"] for e in es}) == 1


def test_resume_reruns_only_failed_and_skipped(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, llms = build(fail_times=1)
    r1 = o.run(dag(), log=log)
    assert (r1["x"].status, r1["y"].status, r1["z"].status) == ("ok", "failed", "skipped")
    o2, llms2 = build()
    r2 = o2.run(dag(), log=log, resume=True)
    assert all(r.status == "ok" for r in r2.values())
    assert r2["x"].resumed and r2["x"].output == r1["x"].output
    assert not r2["y"].resumed and not r2["z"].resumed
    assert llms2["a"].calls == 0 and llms2["b"].calls == 1 and llms2["c"].calls == 1


def test_resume_is_noop_when_everything_done(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, _ = build()
    r1 = o.run(dag(), log=log)
    o2, llms2 = build()
    r2 = o2.run(dag(), log=log, resume=True)
    assert all(l.calls == 0 for l in llms2.values())
    assert {k: v.output for k, v in r2.items()} == {k: v.output for k, v in r1.items()}
    assert all(r.resumed for r in r2.values())


def test_edited_prompt_reruns_task_but_not_siblings(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, _ = build()
    o.run(dag(), log=log)
    o2, llms2 = build()
    o2.run(dag(prompt_c="C2 {x}{y}"), log=log, resume=True)
    assert llms2["a"].calls == 0 and llms2["b"].calls == 0 and llms2["c"].calls == 1


def test_resume_without_log_flag_does_not_replay(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, _ = build()
    o.run(dag(), log=log)
    o2, llms2 = build()
    o2.run(dag(), log=log)  # resume defaults to False
    assert llms2["a"].calls == 1


def test_downstream_uses_resumed_output(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    o, _ = build(fail_times=1)
    r1 = o.run(dag(), log=log)

    seen = []
    class Spy:
        def complete(self, m):
            seen.append(m[-1]["content"]); return "done"
    o2 = Orchestrator([Agent("a", Counting(), ToolRegistry(), system="a"),
                       Agent("b", Counting(), ToolRegistry(), system="b"),
                       Agent("c", Spy(), ToolRegistry(), system="c")])
    o2.run(dag(), log=log, resume=True)
    assert r1["x"].output in seen[0]    # templated from the replayed result


def test_truncated_trailing_line_is_ignored(tmp_path):
    p = tmp_path / "r.jsonl"
    log = RunLog(str(p))
    o, _ = build()
    o.run(dag(), log=log)
    with open(p, "a") as f:
        f.write('{"run": "zz", "task": "x", "sta')
    assert len(log.entries()) == 3
    o2, llms2 = build()
    o2.run(dag(), log=log, resume=True)
    assert llms2["a"].calls == 0


def test_runs_summary_and_cli_and_compact(tmp_path, capsys):
    p = str(tmp_path / "r.jsonl")
    log = RunLog(p)
    o, _ = build(fail_times=1)
    o.run(dag(), log=log)
    o2, _ = build()
    o2.run(dag(), log=log, resume=True)
    runs = log.runs()
    assert [r["tasks"] for r in runs] == [3, 3]
    assert runs[0]["failed"] == 1 and runs[0]["skipped"] == 1
    assert runs[1]["resumed"] == 1 and runs[1]["ok"] == 3
    assert main(["log", p, "--compact"]) == 0
    out = capsys.readouterr().out
    assert "dropped" in out and "run" in out
    # after compaction: x ok (run1), y failed(run1) superseded? different records keyed by (task, fp)
    es = log.entries()
    assert all(not e["resumed"] for e in es)
    assert {e["task"] for e in es} == {"x", "y", "z"}


def test_concurrent_logging_produces_valid_lines(tmp_path):
    log = RunLog(str(tmp_path / "r.jsonl"))
    ags = [Agent("a", Counting(), ToolRegistry(), system="a")]
    tasks = [Task(f"t{i}", "a", f"p{i}") for i in range(30)]
    Orchestrator(ags, max_workers=8).run(tasks, log=log)
    with open(log.path) as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 30
