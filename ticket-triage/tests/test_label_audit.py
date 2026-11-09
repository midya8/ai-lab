from triage.data import generate
from triage.label_audit import audit_labels, flip_labels


def test_flip_labels_count_and_change():
    labels = ["a", "b", "c"] * 20
    noisy, fl = flip_labels(labels, 0.1, seed=1)
    assert len(fl) == 6
    assert all(noisy[i] != labels[i] for i in fl)
    assert sum(a != b for a, b in zip(labels, noisy)) == 6


def test_audit_finds_injected_errors():
    d = generate(500, seed=3)
    texts = [x["text"] for x in d]
    clean = [x["category"] for x in d]
    noisy, fl = flip_labels(clean, 0.08, seed=3)
    res = audit_labels(texts, noisy)
    found = {r["index"] for r in res}
    prec = len(found & set(fl)) / max(1, len(found))
    rec = len(found & set(fl)) / len(fl)
    assert prec > 0.7 and rec > 0.5
    fixed = sum(1 for r in res if r["index"] in set(fl) and r["suggested"] == clean[r["index"]])
    assert fixed >= 0.7 * len(found & set(fl))


def test_sorted_and_top():
    d = generate(300, seed=4)
    noisy, _ = flip_labels([x["category"] for x in d], 0.1, seed=4)
    res = audit_labels([x["text"] for x in d], noisy, top=5)
    assert len(res) <= 5
    q = [r["quality"] for r in res]
    assert q == sorted(q)
