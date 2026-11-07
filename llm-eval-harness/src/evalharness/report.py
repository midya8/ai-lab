def compare(baseline, candidate, threshold=0.0):
    """Per-case regression diff between two runs of the same suite."""
    base = {r["id"]: r["score"] for r in baseline["results"]}
    regressions, improvements = [], []
    for r in candidate["results"]:
        if r["id"] in base:
            d = r["score"] - base[r["id"]]
            if d < -threshold:
                regressions.append((r["id"], base[r["id"]], r["score"]))
            elif d > threshold:
                improvements.append((r["id"], base[r["id"]], r["score"]))
    delta = candidate["mean_score"] - baseline["mean_score"]
    return {"delta": delta, "regressions": regressions,
            "improvements": improvements, "passed": not regressions}


def render_markdown(run, diff=None):
    L = [f"# {run['suite']} on `{run['backend']}`", "",
         f"- cases: {run['n']}", f"- mean score: **{run['mean_score']:.3f}**",
         f"- errors: {run['errors']}", ""]
    if run["by_tag"]:
        L += ["| tag | score |", "|---|---|"]
        L += [f"| {t} | {v:.3f} |" for t, v in run["by_tag"].items()] + [""]
    if diff:
        L.append(f"Delta vs baseline: {diff['delta']:+.3f} "
                 f"({'PASS' if diff['passed'] else 'REGRESSION'})")
        for i, a, b in diff["regressions"]:
            L.append(f"- regressed `{i}`: {a:.2f} -> {b:.2f}")
    fails = [r for r in run["results"] if r["score"] < 1]
    if fails:
        L += ["", "## Failing cases"]
        L += [f"- `{r['id']}` ({r['score']:.2f}): {r['output'][:80]!r}" for r in fails]
    return "\n".join(L) + "\n"
