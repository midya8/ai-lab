import csv
import json

import numpy as np
import pytest

from datadoctor.__main__ import main
from datadoctor.audit import audit, render_markdown
from datadoctor.evaluate import auroc, average_precision, compare_detectors, mask_scores, precision_at_k
from datadoctor.io import load_csv, write_csv_without
from datadoctor.synth import inject_label_noise, make_blobs


def test_metrics_on_known_rankings():
    truth = np.array([True, True, False, False])
    perfect = np.array([0.9, 0.8, 0.2, 0.1])
    worst = -perfect
    assert precision_at_k(perfect, truth) == 1.0 and precision_at_k(worst, truth) == 0.0
    assert average_precision(perfect, truth) == 1.0
    assert auroc(perfect, truth) == 1.0 and auroc(worst, truth) == 0.0
    assert auroc(np.zeros(4), truth) == 0.5  # all ties
    # hand-computed AP: hits at ranks 1 and 3 -> (1/1 + 2/3)/2
    assert average_precision(np.array([4.0, 2.0, 3.0, 1.0]), truth) == pytest.approx((1 + 2 / 3) / 2)


def test_mask_scores():
    r = mask_scores(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["f1"] == 0.5


@pytest.mark.parametrize("kind", ["symmetric", "pairwise", "asymmetric"])
def test_noise_injection_exact_rate_and_always_wrong(kind):
    y = np.arange(1000) % 5
    yn, fl = inject_label_noise(y, 0.2, kind=kind, seed=3)
    assert fl.sum() == 200
    assert (yn[fl] != y[fl]).all() and (yn[~fl] == y[~fl]).all()


def test_pairwise_noise_is_systematic():
    y = np.arange(500) % 4
    yn, fl = inject_label_noise(y, 0.3, kind="pairwise", seed=0)
    assert ((yn[fl] - y[fl]) % 4 == 1).all()


def test_detection_beats_random_by_a_wide_margin():
    X, y = make_blobs(n=900, seed=1)
    yn, fl = inject_label_noise(y, 0.2, seed=1)
    r = compare_detectors(X, yn, fl, seed=1)
    assert r["ranking"]["CV margin"]["auroc"] > 0.93
    assert r["ranking"]["random"]["auroc"] < 0.6
    assert r["masks"]["both"]["precision"] > 0.75
    assert abs(r["estimated_noise_rate"] - 0.2) < 0.08


def _write(path, X, y, texts=None):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"f{i}" for i in range(X.shape[1])] + (["note"] if texts else []) + ["label"])
        for i in range(len(y)):
            w.writerow(list(X[i]) + ([texts[i]] if texts else []) + [f"c{y[i]}"])


def test_csv_roundtrip_and_audit_report(tmp_path):
    X, y = make_blobs(n=500, n_classes=3, sep=1.6, seed=0)
    yn, fl = inject_label_noise(y, 0.15, seed=0)
    p = tmp_path / "d.csv"
    _write(p, X, yn, texts=[f"row {i} some words here to compare" for i in range(500)])
    X2, y2, names, texts, feats = load_csv(str(p), "label", "note")
    assert X2.shape == (500, 12) and names == ["c0", "c1", "c2"] and len(texts) == 500
    assert feats[0] == "f0" and "note" not in feats
    r = audit(X2, y2, class_names=names, texts=texts)
    flagged = {it["row"] for it in r["label_issues"]}
    assert len(flagged & set(np.flatnonzero(fl))) / max(1, len(flagged)) > 0.7
    md = render_markdown(r)
    assert "# Dataset audit" in md and "## Label issues" in md
    json.dumps(r)  # serialisable
    out = tmp_path / "clean.csv"
    n = write_csv_without(str(p), str(out), flagged)
    assert n == 500 - len(flagged)


def test_cli_audit_markdown_json_and_fail_gate(tmp_path, capsys):
    X, y = make_blobs(n=300, n_classes=3, seed=0)
    yn, _ = inject_label_noise(y, 0.2, seed=0)
    p = tmp_path / "d.csv"
    _write(p, X, yn)
    assert main(["audit", str(p), "--label", "label"]) == 0
    assert "Dataset audit" in capsys.readouterr().out
    assert main(["audit", str(p), "--label", "label", "--fail-above", "0.01"]) == 1
    capsys.readouterr()
    clean = tmp_path / "clean.csv"
    assert main(["audit", str(p), "--label", "label", "--json", "--write-clean", str(clean)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n"] == 300 and data["n_label_issues"] > 0
    assert sum(1 for _ in open(clean)) == 1 + 300 - len(data["label_issues"])


def test_cli_demo_prints_both_tables(capsys):
    assert main(["demo", "--n", "400", "--noise", "0.2"]) == 0
    out = capsys.readouterr().out
    assert "CV margin" in out and "prune_by_noise_rate" in out


def test_load_csv_errors(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("a,b\n1,x\n2,y\n")
    with pytest.raises(ValueError, match="not in"):
        load_csv(str(p), "missing")
    q = tmp_path / "t.csv"
    q.write_text("a,b\nfoo,x\nbar,y\n")
    with pytest.raises(ValueError, match="no numeric"):
        load_csv(str(q), "b")
    r = tmp_path / "empty.csv"
    r.write_text("a,b\n")
    with pytest.raises(ValueError, match="empty"):
        load_csv(str(r), "b")
