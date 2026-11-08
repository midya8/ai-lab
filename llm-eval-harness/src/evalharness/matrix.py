"""Multi-model comparison matrix: run one suite across several backends."""
from .runner import run_suite


def run_matrix(backends, suite, workers=4):
    """Run `suite` against each backend; returns {'suite', 'runs': [run, ...]}."""
    names = [b.name for b in backends]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate backend names: {names}")
    return {"suite": suite["name"],
            "runs": [run_suite(b, suite, workers) for b in backends]}


def matrix_table(matrix):
    """Rows per case, columns per backend, plus per-backend means and a winner."""
    runs = matrix["runs"]
    names = [r["backend"] for r in runs]
    ids = [x["id"] for x in runs[0]["results"]] if runs else []
    by = [{x["id"]: x["score"] for x in r["results"]} for r in runs]
    rows = [{"id": i, "scores": [d.get(i, 0.0) for d in by]} for i in ids]
    means = [r["mean_score"] for r in runs]
    best = max(means) if means else 0.0
    winners = [n for n, m in zip(names, means) if m == best]
    contested = [r["id"] for r in rows if len(set(r["scores"])) > 1]
    return {"backends": names, "rows": rows, "means": means,
            "winners": winners, "contested": contested}


def tag_table(matrix):
    """Mean score per tag per backend, plus the best backend(s) for each tag."""
    runs = matrix["runs"]
    names = [r["backend"] for r in runs]
    per = []
    for r in runs:
        acc = {}
        for x in r["results"]:
            for t in x.get("tags", []):
                acc.setdefault(t, []).append(x["score"])
        per.append({t: sum(v) / len(v) for t, v in acc.items()})
    tags = sorted({t for d in per for t in d})
    rows = []
    for t in tags:
        scores = [d.get(t, 0.0) for d in per]
        top = max(scores)
        rows.append({"tag": t, "scores": scores,
                     "best": [n for n, s in zip(names, scores) if s == top]})
    return {"backends": names, "rows": rows}


def render_matrix_markdown(matrix):
    t = matrix_table(matrix)
    out = [f"# Matrix: {matrix['suite']}", "",
           "| case | " + " | ".join(t["backends"]) + " |",
           "|---|" + "---|" * len(t["backends"])]
    for r in t["rows"]:
        out.append(f"| {r['id']} | " + " | ".join(f"{s:.2f}" for s in r["scores"]) + " |")
    out.append("| **mean** | " + " | ".join(f"**{m:.3f}**" for m in t["means"]) + " |")
    out += ["", f"Best: {', '.join(t['winners'])}. "
            f"Cases where models disagree: {len(t['contested'])}/{len(t['rows'])}."]
    tt = tag_table(matrix)
    if tt["rows"]:
        out += ["", "## By tag", "", "| tag | " + " | ".join(tt["backends"]) + " | best |",
                "|---|" + "---|" * (len(tt["backends"]) + 1)]
        for r in tt["rows"]:
            out.append(f"| {r['tag']} | " + " | ".join(f"{x:.2f}" for x in r["scores"])
                       + f" | {', '.join(r['best'])} |")
    return "\n".join(out)
