import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from triage.tags import TagClassifier, generate_tagged, multilabel_metrics, TAGS


def split():
    d = generate_tagged(900, seed=3)
    return d[:600], d[600:750], d[750:]


def test_generator_deterministic_and_tags_in_text():
    a, b = generate_tagged(50, seed=1), generate_tagged(50, seed=1)
    assert a == b
    assert any(r["tags"] for r in a) and any(not r["tags"] for r in a)


def test_metrics_known_values():
    m = multilabel_metrics([[1, 0], [0, 1]], [[1, 0], [1, 1]])
    assert m["exact_match"] == 0.5
    assert abs(m["hamming_loss"] - 0.25) < 1e-9
    assert abs(m["micro_f1"] - 0.8) < 1e-9


def test_metrics_empty_tag_is_zero_not_nan():
    m = multilabel_metrics([[0], [0]], [[0], [0]])
    assert m["macro_f1"] == 0.0


def test_classifier_learns_tags():
    tr, va, te = split()
    c = TagClassifier().fit([r["text"] for r in tr], [r["tags"] for r in tr])
    c.tune_thresholds([r["text"] for r in va], [r["tags"] for r in va])
    P = c.predict([r["text"] for r in te])
    m = multilabel_metrics(c._Y([r["tags"] for r in te]), P)
    assert m["micro_f1"] > 0.9 and m["exact_match"] > 0.8


def test_proba_shape_and_range():
    tr, _, te = split()
    c = TagClassifier(epochs=50).fit([r["text"] for r in tr], [r["tags"] for r in tr])
    P = c.predict_proba([r["text"] for r in te])
    assert P.shape == (len(te), len(TAGS)) and P.min() >= 0 and P.max() <= 1


def test_predict_tags_single_ticket():
    tr, va, _ = split()
    c = TagClassifier().fit([r["text"] for r in tr], [r["tags"] for r in tr])
    c.tune_thresholds([r["text"] for r in va], [r["tags"] for r in va])
    t = c.predict_tags("The app crashes when I open profile on Android and it is extremely slow")
    assert "performance" in t and "security" not in t and "enterprise" not in t


def test_tuned_threshold_not_worse_than_default_on_validation():
    tr, va, _ = split()
    c = TagClassifier().fit([r["text"] for r in tr], [r["tags"] for r in tr])
    Y = c._Y([r["tags"] for r in va])
    before = multilabel_metrics(Y, c.predict([r["text"] for r in va]))["micro_f1"]
    c.tune_thresholds([r["text"] for r in va], [r["tags"] for r in va])
    after = multilabel_metrics(Y, c.predict([r["text"] for r in va]))["micro_f1"]
    assert after >= before - 1e-9
