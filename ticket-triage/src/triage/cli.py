import argparse
from .data import generate
from .calibration import cross_validate, expected_calibration_error, CalibratedClassifier
from .urgency import Triage, UrgencyClassifier, precision_recall_f1
from .model import TicketClassifier, confusion, macro_f1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/evaluate ticket triage and classify text")
    ap.add_argument("text", nargs="?", help="ticket text to classify")
    ap.add_argument("--cv", type=int, default=0, help="run k-fold cross-validation")
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
    if a.text:
        print(Triage().fit(tr).route(a.text))


if __name__ == "__main__":
    main()
