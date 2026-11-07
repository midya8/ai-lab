import json
import time
from concurrent.futures import ThreadPoolExecutor

from .scorers import score


def load_suite(path):
    with open(path) as f:
        suite = json.load(f)
    for key in ("name", "cases"):
        if key not in suite:
            raise ValueError(f"suite missing {key!r}")
    return suite


def _one(backend, case, template, scorer):
    prompt = template.format(input=case["input"]) if template else case["input"]
    t = time.perf_counter()
    try:
        out, err = backend.generate(prompt), None
    except Exception as e:  # a failing call is a failed case, not a crash
        out, err = "", repr(e)
    s = score(case.get("scorer", scorer), out, case) if err is None else 0.0
    return {"id": case["id"], "output": out, "score": s, "error": err,
            "latency_ms": (time.perf_counter() - t) * 1000,
            "tags": case.get("tags", [])}


def run_suite(backend, suite, workers=4):
    tpl, default = suite.get("template"), suite.get("scorer", "exact")
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda c: _one(backend, c, tpl, default), suite["cases"]))
    n = len(results)
    by_tag = {}
    for r in results:
        for t in r["tags"]:
            by_tag.setdefault(t, []).append(r["score"])
    return {"suite": suite["name"], "backend": backend.name, "n": n,
            "mean_score": sum(r["score"] for r in results) / n if n else 0.0,
            "errors": sum(r["error"] is not None for r in results),
            "by_tag": {t: sum(v) / len(v) for t, v in sorted(by_tag.items())},
            "results": results}
