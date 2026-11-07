import argparse
from .data import generate
from .model import TicketClassifier, confusion, macro_f1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train/evaluate ticket triage and classify text")
    ap.add_argument("text", nargs="?", help="ticket text to classify")
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
    if a.text:
        print(clf.predict_with_confidence(a.text))


if __name__ == "__main__":
    main()
