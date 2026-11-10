"""CLI: calibrate and benchmark speculative decoding on any text file.

python -m specdec FILE [--target-order 4] [--draft-order 1] [--cost-ratio 0.1] [--n 2000]

Byte-level n-gram target/draft pair fit on FILE. Reports the calibrated
alpha, recommended k, predicted speedup, and the measured acceptance rate and
tokens per target call from an actual run.
"""
import argparse
import json
import sys

import numpy as np

from . import NgramLM, recommend_k, speculative_generate, expected_speedup


def run(data, target_order=4, draft_order=1, cost_ratio=0.1, n=2000, seed=0, k_max=16):
    toks = list(data)
    if len(toks) < 50:
        raise ValueError("need at least 50 bytes of input")
    target = NgramLM(256, target_order).fit(toks)
    draft = NgramLM(256, draft_order, alpha=0.5).fit(toks)
    step = max(1, len(toks) // 200)
    ctxs = [toks[max(0, i - 8):i] or [toks[0]] for i in range(step, len(toks), step)]
    k, alpha, pred = recommend_k(target, draft, ctxs, cost_ratio, k_max)
    st = {}
    speculative_generate(target, draft, toks[:4], n, k, np.random.default_rng(seed), st)
    return {
        "alpha_calibrated": round(alpha, 4), "k": k,
        "predicted_speedup": round(pred, 3),
        "alpha_measured": round(st["accepted"] / max(1, st["proposed"]), 4),
        "tokens_per_target_call": round(n / st["target_calls"], 3),
        "predicted_speedup_at_measured_alpha": round(expected_speedup(
            min(0.999, st["accepted"] / max(1, st["proposed"])), k, cost_ratio), 3),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(prog="specdec", description=__doc__.split("\n")[0])
    ap.add_argument("file")
    ap.add_argument("--target-order", type=int, default=4)
    ap.add_argument("--draft-order", type=int, default=1)
    ap.add_argument("--cost-ratio", type=float, default=0.1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    with open(a.file, "rb") as f:
        data = f.read()
    try:
        res = run(data, a.target_order, a.draft_order, a.cost_ratio, a.n, a.seed)
    except ValueError as e:
        print("error:", e, file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
