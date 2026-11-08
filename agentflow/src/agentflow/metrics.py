"""Run metrics: critical path, total work and parallel speedup of a task graph."""
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RunMetrics:
    total_work: float          # sum of task durations (sequential cost)
    critical_path: float       # longest dependency chain by duration
    critical_tasks: List[str]  # task ids on that chain
    parallelism: float         # total_work / critical_path (ideal speedup)
    failed: int
    skipped: int
    retries: int


def compute_metrics(tasks, results: Dict[str, "object"]) -> RunMetrics:
    by_id = {t.id: t for t in tasks}
    best: Dict[str, float] = {}
    prev: Dict[str, str] = {}

    def finish(tid: str) -> float:
        if tid in best:
            return best[tid]
        t = by_id[tid]
        base, via = 0.0, None
        for d in t.deps:
            f = finish(d)
            if f > base:
                base, via = f, d
        best[tid] = base + results[tid].duration
        if via:
            prev[tid] = via
        return best[tid]

    for tid in by_id:
        finish(tid)
    total = sum(r.duration for r in results.values())
    end = max(best, key=best.get) if best else None
    path: List[str] = []
    while end:
        path.append(end)
        end = prev.get(end)
    path.reverse()
    cp = best[path[-1]] if path else 0.0
    return RunMetrics(
        total_work=total,
        critical_path=cp,
        critical_tasks=path,
        parallelism=(total / cp) if cp > 0 else 1.0,
        failed=sum(r.status == "failed" for r in results.values()),
        skipped=sum(r.status == "skipped" for r in results.values()),
        retries=sum(max(r.attempts - 1, 0) for r in results.values()),
    )
