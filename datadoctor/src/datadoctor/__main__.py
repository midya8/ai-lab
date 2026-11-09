from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from .audit import audit, render_markdown
from .evaluate import cleaning_effect, compare_detectors
from .report import render_html
from .io import load_csv, write_csv_without
from .synth import inject_label_noise, make_blobs


def _cmd_audit(a):
    X, y, names, texts, feats = load_csv(a.csv, a.label, a.text)
    r = audit(X, y, class_names=names, texts=texts, folds=a.folds, repeats=a.repeats, seed=a.seed, method=a.method)
    if a.json:
        print(json.dumps(r, indent=2))
    else:
        md = render_markdown(r)
        print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write(json.dumps(r, indent=2) if a.json else render_markdown(r))
    if a.html:
        with open(a.html, "w") as f:
            f.write(render_html(r))
    if a.write_clean:
        n = write_csv_without(a.csv, a.write_clean, {it["row"] for it in r["label_issues"]})
        print(f"wrote {n} rows to {a.write_clean} (flagged rows removed)", file=sys.stderr)
    return 1 if (a.fail_above is not None and r["n_label_issues"] / r["n"] > a.fail_above) else 0


def _cmd_demo(a):
    X, y = make_blobs(n=a.n, seed=a.seed)
    yn, flipped = inject_label_noise(y, a.noise, kind=a.kind, seed=a.seed)
    res = compare_detectors(X, yn, flipped, repeats=a.repeats, seed=a.seed)
    print(f"{a.n} rows, {a.noise:.0%} {a.kind} label noise ({int(flipped.sum())} flipped)\n")
    print("| detector | P@k | AP | AUROC |\n|---|---|---|---|")
    for k, v in res["ranking"].items():
        print(f"| {k} | {v['p_at_k']:.3f} | {v['ap']:.3f} | {v['auroc']:.3f} |")
    print("\n| rule | flagged | precision | recall | F1 |\n|---|---|---|---|---|")
    for k, v in res["masks"].items():
        print(f"| {k} | {v['flagged']} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="datadoctor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("audit", help="audit a CSV for label errors, duplicates, outliers")
    p.add_argument("csv")
    p.add_argument("--label", required=True)
    p.add_argument("--text", help="optional text column for near-duplicate text detection")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", default="both", choices=["argmax", "confident_joint", "prune_by_class", "prune_by_noise_rate", "both"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--out")
    p.add_argument("--html", help="also write a self-contained HTML report")
    p.add_argument("--write-clean", help="write a copy of the CSV without flagged rows")
    p.add_argument("--fail-above", type=float, help="exit 1 if the flagged share exceeds this fraction (CI gate)")
    p.set_defaults(fn=_cmd_audit)
    d = sub.add_parser("demo", help="inject label noise into synthetic data and score the detectors")
    d.add_argument("--n", type=int, default=1500)
    d.add_argument("--noise", type=float, default=0.2)
    d.add_argument("--kind", default="symmetric", choices=["symmetric", "pairwise", "asymmetric"])
    d.add_argument("--repeats", type=int, default=1)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=_cmd_demo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
