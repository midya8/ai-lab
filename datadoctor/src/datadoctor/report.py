"""Self-contained HTML rendering of an audit result (no external assets, no JS)."""

from __future__ import annotations

from html import escape

_CSS = (
    "body{font:15px/1.5 system-ui,sans-serif;max-width:60rem;margin:2rem auto;padding:0 1rem;color:#222}"
    "table{border-collapse:collapse;margin:.5rem 0}td,th{border:1px solid #ccc;padding:.25rem .6rem;text-align:right}"
    "th:first-child,td:first-child{text-align:left}.bar{background:#4c78a8;height:.8rem;display:inline-block}"
    ".warn{color:#b00020}.note{color:#555}"
)


def _table(head: list[str], rows: list[list[str]]) -> str:
    h = "".join(f"<th>{escape(str(c))}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><tr>{h}</tr>{body}</table>"


def _heat(v: float) -> str:
    a = max(0.0, min(1.0, v))
    return f'<td style="background:rgba(76,120,168,{a:.2f})">{v:.2f}</td>'


def render_html(r: dict, top: int = 25) -> str:
    n = r["n"]
    names = r["class_names"]
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>Dataset audit</title><style>{_CSS}</style></head><body>"]
    out.append("<h1>Dataset audit</h1>")
    out.append(
        f"<p>{n} rows, {r['n_classes']} classes. {r['folds']}-fold cross-validated accuracy against the given labels: "
        f"<b>{r['cv_accuracy']:.3f}</b>.</p>"
    )
    rows = []
    for name, c, p in zip(names, r["class_counts"], r["estimated_true_prior"]):
        w = int(200 * c / max(r["class_counts"]))
        rows.append([escape(name), f'{c} <span class="bar" style="width:{w}px"></span>', f"{p:.1%}"])
    out.append("<h2>Class balance</h2>" + _table(["class", "given", "estimated true share"], rows))
    out.append("<h2>Label issues</h2>")
    out.append(
        f"<p>{r['n_label_issues']} rows ({r['n_label_issues'] / n:.1%}) look mislabelled (rule: {escape(r['method'])}); "
        f"estimated noise rate {r['estimated_noise_rate']:.1%}. <span class='note'>Candidates for human review, not verdicts.</span></p>"
    )
    its = r["label_issues"][:top]
    if its:
        out.append(
            _table(
                ["row", "given", "suggested", "P(given)", "P(suggested)"],
                [[it["row"], escape(it["given"]), escape(it["suggested"]), f"{it['p_given']:.2f}", f"{it['p_suggested']:.2f}"] for it in its],
            )
        )
        if len(r["label_issues"]) > top:
            out.append(f"<p class='note'>{len(r['label_issues']) - top} more not shown.</p>")
    out.append("<h2>Estimated noise matrix</h2><p class='note'>P(given label | true class); rows are true classes.</p>")
    head = "".join(f"<th>{escape(x)}</th>" for x in ["true \\ given"] + names)
    body = "".join(
        f"<tr><td>{escape(names[i])}</td>" + "".join(_heat(v) for v in row) + "</tr>" for i, row in enumerate(r["noise_matrix"])
    )
    out.append(f"<table><tr>{head}</tr>{body}</table>")
    out.append("<h2>Duplicates</h2><ul>")
    out.append(f"<li>exact duplicate groups: {len(r['exact_duplicate_groups'])}</li>")
    nc = len(r["conflicting_duplicate_groups"])
    cls = " class='warn'" if nc else ""
    out.append(f"<li{cls}>groups with conflicting labels: {nc}</li>")
    out.append(f"<li>near-duplicate pairs: {r['n_near_duplicate_pairs']}</li>")
    if "near_duplicate_texts" in r:
        out.append(f"<li>near-duplicate texts: {len(r['near_duplicate_texts'])}</li>")
    out.append("</ul><h2>Outliers</h2>")
    if r["outliers"]:
        out.append(_table(["row", "robust z"], [[o["row"], f"{o['robust_z']:.1f}"] for o in r["outliers"][:top]]))
    else:
        out.append("<p>None above robust z 3.5.</p>")
    out.append("</body></html>")
    return "".join(out)
