"""K-fold CV, expected calibration error, and temperature scaling."""
import numpy as np
from .model import TicketClassifier, softmax, macro_f1


def kfold_indices(n, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def cross_validate(texts, labels, k=5, seed=0, **kw):
    """Returns per-fold accuracy and macro-F1 plus their means."""
    accs, f1s = [], []
    for tr, te in kfold_indices(len(texts), k, seed):
        clf = TicketClassifier(**kw).fit([texts[i] for i in tr], [labels[i] for i in tr])
        pred = clf.predict([texts[i] for i in te])
        true = [labels[i] for i in te]
        accs.append(float(np.mean([p == t for p, t in zip(pred, true)])))
        f1s.append(macro_f1(true, pred, clf.classes))
    return {"acc": accs, "f1": f1s, "acc_mean": float(np.mean(accs)), "f1_mean": float(np.mean(f1s))}


def expected_calibration_error(probs, y_idx, bins=10):
    """ECE: weighted gap between confidence and accuracy over equal-width bins."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y_idx)).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def nll(logits, y_idx, T=1.0):
    p = softmax(logits / T)
    return float(-np.mean(np.log(p[np.arange(len(y_idx)), y_idx] + 1e-12)))


def fit_temperature(logits, y_idx, grid=None):
    """Pick temperature minimizing validation NLL by grid search."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120)) if grid is None else grid
    return float(min(grid, key=lambda T: nll(logits, y_idx, T)))


class CalibratedClassifier:
    """Wraps TicketClassifier; learns a temperature on a held-out split."""

    def __init__(self, val_frac=0.2, seed=0, **kw):
        self.val_frac, self.seed, self.kw = val_frac, seed, kw

    def fit(self, texts, labels):
        n = len(texts)
        idx = np.random.RandomState(self.seed).permutation(n)
        nv = max(1, int(n * self.val_frac))
        va, tr = idx[:nv], idx[nv:]
        self.clf = TicketClassifier(**self.kw).fit([texts[i] for i in tr], [labels[i] for i in tr])
        y = np.array([self.clf.classes.index(labels[i]) for i in va if labels[i] in self.clf.classes])
        keep = [i for i in va if labels[i] in self.clf.classes]
        self.T = fit_temperature(self._logits([texts[i] for i in keep]), y)
        return self

    def _logits(self, texts):
        return self.clf.vec.transform(texts) @ self.clf.W + self.clf.b

    @property
    def classes(self):
        return self.clf.classes

    def predict_proba(self, texts):
        return softmax(self._logits(texts) / self.T)

    def predict(self, texts):
        return [self.classes[i] for i in self.predict_proba(texts).argmax(axis=1)]
