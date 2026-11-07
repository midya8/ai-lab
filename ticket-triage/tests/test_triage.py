import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triage.data import generate
from triage.model import TicketClassifier, TfidfVectorizer, macro_f1, softmax


def split():
    d = generate(600)
    return d[:480], d[480:]


def test_softmax_rows_sum_to_one():
    p = softmax(np.random.randn(4, 3))
    assert np.allclose(p.sum(axis=1), 1)


def test_tfidf_rows_unit_norm():
    v = TfidfVectorizer().fit(["a b c", "a b d", "a c d"])
    X = v.transform(["a b"])
    assert abs(np.linalg.norm(X[0]) - 1) < 1e-9


def test_accuracy_on_holdout():
    tr, te = split()
    clf = TicketClassifier().fit([d["text"] for d in tr], [d["category"] for d in tr])
    pred = clf.predict([d["text"] for d in te])
    true = [d["category"] for d in te]
    assert macro_f1(true, pred, clf.classes) > 0.9


def test_confidence_and_escalation():
    tr, _ = split()
    clf = TicketClassifier().fit([d["text"] for d in tr], [d["category"] for d in tr])
    r = clf.predict_with_confidence("I was charged twice for my plan")
    assert r["label"] == "billing"
    assert clf.predict_with_confidence("zzz qqq", threshold=0.9)["needs_human"]


def test_deterministic_data():
    assert generate(10, 1) == generate(10, 1)
