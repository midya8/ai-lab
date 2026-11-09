"""Fusion quality on a synthetic corpus with two query types (fully synthetic; see README caveats).

Each doc has a topic (dense signal: topic centroid + noise) and a text of topic words drawn from a
per-topic vocabulary plus one rare ID token. Two query kinds:
  semantic: the vector is near the target's topic; the text uses *other* words of that topic (little
            lexical overlap with the target, many same-topic distractors) -> dense should help.
  lookup:   the text holds the target's rare ID; the vector is noisy (weakly informative) -> BM25 helps.
Metrics: MRR@10 and recall@10 of the single relevant doc, per query kind and mixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vecdb import HybridCollection  # noqa: E402


def build(n=1500, dim=32, topics=30, seed=0, dense_noise=1.0):
    rng = np.random.default_rng(seed)
    cent = rng.normal(size=(topics, dim)).astype(np.float32)
    vocab = [[f"t{t}w{j}" for j in range(40)] for t in range(topics)]
    topic = rng.integers(0, topics, n)
    vecs = cent[topic] + dense_noise * rng.normal(size=(n, dim)).astype(np.float32)
    texts = []
    for i in range(n):
        words = list(rng.choice(vocab[topic[i]], 6, replace=False)) + [f"id{i:05d}"]
        texts.append(" ".join(words))
    return rng, cent, vocab, topic, vecs, texts


def make_queries(rng, cent, vocab, topic, texts, nq=300, lookup_vec_noise=3.0):
    qs = []
    n = len(texts)
    for q in range(nq):
        i = int(rng.integers(0, n))
        t = topic[i]
        if q % 2 == 0:
            own = set(texts[i].split())
            pool = [w for w in vocab[t] if w not in own]
            text = " ".join(rng.choice(pool, 3, replace=False))
            vec = cent[t] + 0.5 * rng.normal(size=cent.shape[1]).astype(np.float32)
            qs.append(("semantic", i, vec, text))
        else:
            vec = cent[t] + lookup_vec_noise * rng.normal(size=cent.shape[1]).astype(np.float32)
            qs.append(("lookup", i, vec, f"id{i:05d}"))
    return qs


def evaluate(h, qs, k=10):
    modes = {
        "dense": lambda v, t: [x.id for x in h.col.query(v[None], k)[0]],
        "bm25": lambda v, t: [i for i, _ in h.lexical(t, k)],
        "rrf": lambda v, t: [x.id for x in h.query(v, t, k=k, fusion="rrf")],
        "weighted a=0.5": lambda v, t: [x.id for x in h.query(v, t, k=k, fusion="weighted", alpha=0.5)],
        "weighted a=0.2": lambda v, t: [x.id for x in h.query(v, t, k=k, fusion="weighted", alpha=0.2)],
    }
    out = {}
    for name, fn in modes.items():
        acc = {"semantic": [], "lookup": []}
        for kind, i, v, t in qs:
            ids = fn(v, t)
            tgt = f"d{i}"
            acc[kind].append(1.0 / (ids.index(tgt) + 1) if tgt in ids else 0.0)
        rec = {kd: float(np.mean([x > 0 for x in a])) for kd, a in acc.items()}
        mrr = {kd: float(np.mean(a)) for kd, a in acc.items()}
        out[name] = {"mrr": mrr, "recall": rec,
                     "mrr_all": float(np.mean(acc["semantic"] + acc["lookup"]))}
    return out


def run(seed=0, n=1500, nq=300):
    rng, cent, vocab, topic, vecs, texts = build(n=n, seed=seed)
    h = HybridCollection(vecs.shape[1])
    h.upsert([f"d{i}" for i in range(n)], vecs, texts=texts)
    return evaluate(h, make_queries(rng, cent, vocab, topic, texts, nq=nq))


if __name__ == "__main__":
    res = [run(seed=s) for s in range(3)]
    print("| mode | MRR semantic | MRR lookup | MRR all | recall@10 semantic | recall@10 lookup |")
    print("|---|---|---|---|---|---|")
    for m in res[0]:
        f = lambda g: float(np.mean([r[m][g[0]][g[1]] if len(g) > 1 else r[m][g[0]] for r in res]))
        print(f"| {m} | {f(('mrr','semantic')):.3f} | {f(('mrr','lookup')):.3f} | {f(('mrr_all',)):.3f} "
              f"| {f(('recall','semantic')):.3f} | {f(('recall','lookup')):.3f} |")
