"""Near-duplicate ticket detection: TF-IDF cosine + union-find clustering."""
import math
import re
from collections import Counter

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOK.findall(text.lower())


def _vectors(texts):
    docs = [Counter(_tokens(t)) for t in texts]
    df = Counter(w for d in docs for w in d)
    n = len(texts)
    idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
    vecs = []
    for d in docs:
        v = {w: (1 + math.log(c)) * idf[w] for w, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(w, 0.0) for w, x in a.items())


def find_duplicates(texts, threshold=0.8):
    """Return (i, j, similarity) pairs with i<j and cosine >= threshold, most similar first."""
    vecs = _vectors(texts)
    pairs = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            s = _cos(vecs[i], vecs[j])
            if s >= threshold:
                pairs.append((i, j, round(s, 4)))
    pairs.sort(key=lambda p: -p[2])
    return pairs


def cluster(texts, threshold=0.8):
    """Group tickets into duplicate clusters (size >= 2), each sorted, largest first."""
    parent = list(range(len(texts)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in find_duplicates(texts, threshold):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(len(texts)):
        groups.setdefault(find(i), []).append(i)
    out = [sorted(g) for g in groups.values() if len(g) > 1]
    out.sort(key=lambda g: (-len(g), g[0]))
    return out
