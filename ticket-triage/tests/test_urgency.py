import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triage.data import generate
from triage.urgency import UrgencyClassifier, Triage, precision_recall_f1


def split():
    d = generate(800, seed=1)
    return d[:640], d[640:]


def test_prf_known_values():
    r = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert r == {"precision": 0.5, "recall": 0.5, "f1": 0.5}


def test_prf_no_positives_is_zero():
    assert precision_recall_f1([0, 0], [0, 0])["f1"] == 0.0


def test_urgency_detects_marker_words():
    tr, te = split()
    m = UrgencyClassifier().fit([d["text"] for d in tr], [d["urgent"] for d in tr])
    r = precision_recall_f1([d["urgent"] for d in te], m.predict([d["text"] for d in te]))
    assert r["f1"] > 0.9


def test_probabilities_in_unit_interval():
    tr, _ = split()
    m = UrgencyClassifier(epochs=50).fit([d["text"] for d in tr], [d["urgent"] for d in tr])
    p = m.predict_proba(["anything at all"])
    assert ((p >= 0) & (p <= 1)).all()


def test_tune_threshold_hits_min_recall():
    tr, te = split()
    m = UrgencyClassifier(epochs=100).fit([d["text"] for d in tr], [d["urgent"] for d in tr])
    texts, y = [d["text"] for d in te], [d["urgent"] for d in te]
    m.tune_threshold(texts, y, min_recall=0.98)
    assert precision_recall_f1(y, m.predict(texts))["recall"] >= 0.98


def test_balanced_weight_improves_recall_on_rare_class():
    rng = np.random.default_rng(0)
    d = generate(600, seed=3)
    texts = [x["text"] for x in d]
    y = [bool(x["urgent"]) and rng.random() < 0.3 for x in d]  # very rare + noisy
    plain = UrgencyClassifier(pos_weight=1.0, epochs=100).fit(texts, y)
    bal = UrgencyClassifier(epochs=100).fit(texts, y)
    rp = precision_recall_f1(y, plain.predict(texts))["recall"]
    rb = precision_recall_f1(y, bal.predict(texts))["recall"]
    assert rb >= rp


def test_route_marks_urgent_and_queues():
    tr, _ = split()
    t = Triage().fit(tr)
    r = t.route("The app crashes when I open dashboard production down")
    assert r["urgent"] and r["queue"].endswith(":urgent") and r["needs_human"]
    q = t.route("Please add export to CSV for report")
    assert not q["urgent"] and ":" not in q["queue"]
