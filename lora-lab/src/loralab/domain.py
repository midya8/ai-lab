"""A synthetic 'technical prose' domain used as the adaptation target (task B).

The base model is pretrained on Shakespeare; this grammar is deliberately unlike it,
so adaptation has something real to learn. Generated deterministically from a seed.
"""
import numpy as np

SUBJ = ["the model", "the trainer", "the router", "the encoder", "the decoder", "the agent", "the cache"]
VERB = ["reads", "updates", "scores", "ranks", "caches", "routes", "decodes", "filters"]
OBJ = ["the tokens", "the weights", "the batch", "the gradients", "the queries", "the logits", "the index"]
TAIL = ["in parallel", "after warmup", "with low latency", "under a budget", "before each step", "at every layer"]


def generate_domain_text(n_lines=120, seed=0):
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    lines = [f"{pick(SUBJ)} {pick(VERB)} {pick(OBJ)} {pick(TAIL)}." for _ in range(n_lines)]
    return "\n".join(lines) + "\n"
