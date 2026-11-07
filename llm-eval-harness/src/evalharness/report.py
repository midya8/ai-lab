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


def render_html(run, diff=None):
    """Self-contained HTML report (no external assets); all text is escaped."""
    from html import escape as e
    reg = {i for i, _, _ in diff["regressions"]} if diff else set()
    def bar(s):
        col = "#2da44e" if s >= 1 else "#d29922" if s > 0 else "#cf222e"
        return (f'<div class="bar"><span style="width:{s*100:.0f}%;'
                f'background:{col}"></span></div>')
    rows = "".join(
        f'<tr class="{"reg" if r["id"] in reg else ""}"><td>{e(str(r["id"]))}</td>'
        f'<td>{bar(r["score"])}{r["score"]:.2f}</td><td>{r["latency_ms"]:.1f} ms</td>'
        f'<td><code>{e(r["output"][:200])}</code>{" <b>ERR " + e(r["error"]) + "</b>" if r["error"] else ""}</td></tr>'
        for r in run["results"])
    tags = "".join(f'<tr><td>{e(t)}</td><td>{bar(v)}{v:.3f}</td></tr>'
                   for t, v in run["by_tag"].items())
    verdict = ""
    if diff:
        verdict = (f'<p class="{"ok" if diff["passed"] else "bad"}">Delta vs baseline '
                   f'{diff["delta"]:+.3f}: {"PASS" if diff["passed"] else "REGRESSION"}</p>')
    css = ("body{font:14px system-ui;max-width:900px;margin:2em auto}table{border-collapse:collapse;width:100%}"
           "td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left}.bar{display:inline-block;width:80px;"
           "height:8px;background:#eee;margin-right:6px}.bar span{display:block;height:8px}.reg{background:#ffebe9}"
           ".ok{color:#2da44e}.bad{color:#cf222e}")
    return (f"<!doctype html><meta charset=utf-8><title>{e(run['suite'])}</title><style>{css}</style>"
            f"<h1>{e(run['suite'])} on <code>{e(run['backend'])}</code></h1>"
            f"<p>{run['n']} cases &middot; mean <b>{run['mean_score']:.3f}</b> &middot; {run['errors']} errors</p>"
            f"{verdict}<h2>By tag</h2><table>{tags}</table><h2>Cases</h2>"
            f"<table><tr><th>id</th><th>score</th><th>latency</th><th>output</th></tr>{rows}</table>")
