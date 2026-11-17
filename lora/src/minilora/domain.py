"""A synthetic 'technical prose' domain used as the adaptation target (task B).

The base model is pretrained on Shakespeare; this grammar is deliberately unlike it,
so adaptation has something real to learn. Generated deterministically from a seed.
"""
import numpy as np

SUBJ = ["the model", "the trainer", "the router", "the encoder", "the decoder", "the agent", "the cache"]
VERB = ["reads", "updates", "scores", "ranks", "caches", "routes", "decodes", "filters"]
OBJ = ["the tokens", "the weights", "the batch", "the gradients", "the queries", "the logits", "the index"]
TAIL = ["in parallel", "after warmup", "with low latency", "under a budget", "before each step", "at every layer"]


# A second, deliberately different grammar (kitchen instructions) so a hub can hold two domains.
K_SUBJ = ["the cook", "the baker", "the chef", "she", "he", "the sous chef"]
K_VERB = ["slices", "simmers", "whisks", "roasts", "folds", "seasons", "chops", "tastes"]
K_OBJ = ["the onions", "the dough", "the broth", "the butter", "the garlic", "the sauce", "the eggs"]
K_TAIL = ["until golden", "for ten minutes", "with a pinch of salt", "over low heat", "then lets it rest", "on a warm plate"]

DOMAINS = {
    "tech": (SUBJ, VERB, OBJ, TAIL),
    "kitchen": (K_SUBJ, K_VERB, K_OBJ, K_TAIL),
}


def generate_domain_text(n_lines=120, seed=0, domain="tech"):
    """Deterministic template text for ``domain`` ('tech' by default, or 'kitchen')."""
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    subj, verb, obj, tail = DOMAINS[domain]
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    lines = [f"{pick(subj)} {pick(verb)} {pick(obj)} {pick(tail)}." for _ in range(n_lines)]
    return "\n".join(lines) + "\n"
