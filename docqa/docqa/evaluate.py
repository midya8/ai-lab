"""Evaluate answering and abstention on a labelled question set."""
from __future__ import annotations

import json
from pathlib import Path


def run_eval(qa, path: str | Path) -> dict:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    ans_ok = ans_n = abst_ok = abst_n = 0
    failures = []
    for it in items:
        r = qa.ask(it["question"])
        if it.get("answerable", True):
            ans_n += 1
            good = r.answered and all(
                any(m.lower() in r.text.lower() for m in [x]) for x in it.get("must_contain", [])
            ) and (not it.get("source") or any(it["source"] in e.citation for e in r.evidence))
            ans_ok += good
        else:
            abst_n += 1
            good = not r.answered
            abst_ok += good
        if not good:
            failures.append(it["question"])
    return {
        "answerable": ans_n, "answered_correctly": ans_ok,
        "unanswerable": abst_n, "correctly_abstained": abst_ok,
        "answer_rate": ans_ok / ans_n if ans_n else 0.0,
        "abstain_rate": abst_ok / abst_n if abst_n else 0.0,
        "failures": failures,
    }
