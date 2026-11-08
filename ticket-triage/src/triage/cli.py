import argparse
from .data import generate
from .calibration import cross_validate, expected_calibration_error, CalibratedClassifier
from .urgency import Triage, UrgencyClassifier, precision_recall_f1
from .active import compare as active_compare, review_queue
from .tags import TagClassifier, generate_tagged, multilabel_metrics, TAGS
from .model import TicketClassifier, confusion, macro_f1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/evaluate ticket triage and classify text")
    ap.add_argument("text", nargs="?", help="ticket text to classify")
    ap.add_argument("--cv", type=int, default=0, help="run k-fold cross-validation")
    ap.add_argument("--active", action="store_true", help="active-learning demo: strategy learning curves + review queue")
    ap.add_argument("--tags", action="store_true", help="multi-label tag demo (mobile/performance/security/enterprise)")
    ap.add_argument("--n", type=int, default=800)
    a = ap.parse_args(argv)
    data = generate(a.n)
    cut = int(len(data) * 0.8)
    tr, te = data[:cut], data[cut:]
    clf = TicketClassifier().fit([d["text"] for d in tr], [d["category"] for d in tr])
    pred = clf.predict([d["text"] for d in te])
    true = [d["category"] for d in te]
    acc = sum(p == t for p, t in zip(pred, true)) / len(te)
    print(f"accuracy={acc:.3f} macro_f1={macro_f1(true, pred, clf.classes):.3f}")
    print(clf.classes)
    print(confusion(true, pred, clf.classes))
    if a.cv:
        r = cross_validate([d["text"] for d in data], [d["category"] for d in data], k=a.cv)
        print(f"{a.cv}-fold acc={r['acc_mean']:.3f} f1={r['f1_mean']:.3f}")
    cal = CalibratedClassifier().fit([d["text"] for d in tr], [d["category"] for d in tr])
    yi = [clf.classes.index(t) for t in true]
    texts = [d["text"] for d in te]
    print(f"ECE raw={expected_calibration_error(clf.predict_proba(texts), yi):.3f} "
          f"calibrated={expected_calibration_error(cal.predict_proba(texts), yi):.3f} (T={cal.T:.2f})")
    um = UrgencyClassifier().fit([d["text"] for d in tr], [d["urgent"] for d in tr])
    r = precision_recall_f1([d["urgent"] for d in te], um.predict(texts))
    print(f"urgency precision={r['precision']:.3f} recall={r['recall']:.3f} f1={r['f1']:.3f}")
    if a.active:
        res = active_compare(data[:300], data[300:500], seeds=(0, 1, 2, 3, 4))
        for name, curve in res.items():
            print(f"{name:17s}" + " ".join(f"{n}:{acc:.3f}" for n, acc in curve))
        small = TicketClassifier().fit([d["text"] for d in data[:24]], [d["category"] for d in data[:24]])
        for r in review_queue(small, [d["text"] for d in data[24:300]], k=5, strategy="entropy"):
            print(f"review #{r['index']} ({r['predicted']} {r['confidence']:.2f}): {r['text']}")
    if a.tags:
        td = generate_tagged(a.n)
        c1, c2 = int(len(td) * 0.6), int(len(td) * 0.8)
        f = lambda rows: ([r["text"] for r in rows], [r["tags"] for r in rows])
        tc = TagClassifier().fit(*f(td[:c1]))
        tc.tune_thresholds(*f(td[c1:c2]))
        m = multilabel_metrics(tc._Y(f(td[c2:])[1]), tc.predict(f(td[c2:])[0]))
        print(f"tags micro_f1={m['micro_f1']:.3f} macro_f1={m['macro_f1']:.3f} hamming={m['hamming_loss']:.3f} exact={m['exact_match']:.3f}")
        print(" ".join(f"{t}:{x:.2f}" for t, x in zip(TAGS, m["per_tag_f1"])))
        if a.text:
            print("tags:", tc.predict_tags(a.text))
            from .routing import TaggedTriage
            print("tagged route:", TaggedTriage().fit(td[:c1], td[c1:c2]).route(a.text))
    if a.text:
        print(Triage().fit(tr).route(a.text))


if __name__ == "__main__":
    main()
