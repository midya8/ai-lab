"""Cross-domain transfer: train a tokenizer on each corpus, score it on every corpus.

Cell [i][j] is bytes/token of the tokenizer trained on corpus i, measured on the held-out
tail of corpus j. The diagonal is in-domain; the gap to it is the transfer penalty.
"""
from __future__ import annotations

from .bpe import train_fast


def _split(text: str, holdout: float):
    cut = int(len(text) * (1 - holdout))
    return text[:cut], text[cut:]


def transfer_matrix(corpora: dict[str, str], vocab_size: int = 400, holdout: float = 0.2):
    if not 0 < holdout < 1:
        raise ValueError("holdout must be in (0, 1)")
    if len(corpora) < 2:
        raise ValueError("need at least two corpora")
    names = list(corpora)
    parts = {n: _split(corpora[n], holdout) for n in names}
    toks = {n: train_fast(parts[n][0], vocab_size) for n in names}
    matrix = []
    for i in names:
        row = []
        for j in names:
            held = parts[j][1]
            n_tok = len(toks[i].encode(held))
            row.append(len(held.encode("utf-8")) / max(n_tok, 1))
        matrix.append(row)
    return names, matrix


def transfer_penalty(names, matrix):
    """Per trained-on corpus: mean relative bytes/token loss off-domain vs the best tokenizer."""
    out = {}
    for ci, n in enumerate(names):
        losses = []
        for cj in range(len(names)):
            if cj == ci:
                continue
            best = max(matrix[r][cj] for r in range(len(names)))
            losses.append(1 - matrix[ci][cj] / best)
        out[n] = sum(losses) / len(losses)
    return out


def format_matrix(names, matrix) -> str:
    head = "| trained on \\ scored on | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)
    rows = [f"| {n} | " + " | ".join(f"{v:.2f}" for v in r) + " |" for n, r in zip(names, matrix)]
    return "\n".join([head, sep, *rows])
