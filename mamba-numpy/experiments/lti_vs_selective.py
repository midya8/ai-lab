"""Selective copying: fixed (LTI) vs input-dependent (selective) SSM blocks, same size otherwise.

usage: python lti_vs_selective.py {selective,lti} [out.json]
For each noise length, 3 seeds, 2500 steps, 4 data tokens out of 4 symbols (chance = 0.25 per token).
Also scores the trained model on a LONGER, unseen noise length (train length x 2) without retraining.
"""
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mamba import Mamba, score, selective_copy  # noqa: E402
from mamba.train import fit  # noqa: E402

mode = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"lti_vs_selective_{mode}.json"
LENGTHS, SEEDS, STEPS, N_DATA, VOCAB = [16, 32, 64], [0, 1, 2], 2500, 4, 4

results = []
for L in LENGTHS:
    for seed in SEEDS:
        mk = functools.partial(selective_copy, seq_len=L, n_data=N_DATA, vocab=VOCAB)
        m = Mamba(vocab=VOCAB + 2, d_model=16, d_state=8, n_layer=2, selective=(mode == "selective"), seed=seed)
        t0 = time.time()
        hist = fit(m, lambda n, r: mk(n, rng=r), steps=STEPS, batch=32, lr=3e-3, seed=seed, log_every=STEPS // 5)
        test = selective_copy(1000, L, N_DATA, VOCAB, np.random.default_rng(777 + seed))
        tok, exact = score(m.forward(test[0]), test[1])
        long = selective_copy(1000, 2 * L, N_DATA, VOCAB, np.random.default_rng(888 + seed))
        ltok, lexact = score(m.forward(long[0]), long[1])
        r = dict(mode=mode, seq_len=L, seed=seed, tok=tok, exact=exact, long_len=2 * L, long_tok=ltok,
                 long_exact=lexact, params=m.n_params(), seconds=time.time() - t0,
                 curve=[(h["step"], h["tok_acc"]) for h in hist])
        results.append(r)
        print(f"{mode:9s} L={L:3d} seed={seed} tok={tok:.3f} exact={exact:.3f} | 2L tok={ltok:.3f} exact={lexact:.3f} "
              f"({r['seconds']:.0f}s)", flush=True)
        Path(out).write_text(json.dumps(results, indent=1))
