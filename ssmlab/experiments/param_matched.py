"""Control for lti_vs_selective.py: the LTI model there has ~28% fewer parameters (no W_dt/W_B/W_C).
Here the LTI model is widened to d_model=20 (7,580 params vs the selective model's 7,344), noise length 32.

usage: python param_matched.py SEED [SEED ...]
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssmlab import Mamba, score, selective_copy  # noqa: E402
from ssmlab.train import fit  # noqa: E402

L, N_DATA, VOCAB, STEPS = 32, 4, 4, 2500
for seed in map(int, sys.argv[1:]):
    mk = functools.partial(selective_copy, seq_len=L, n_data=N_DATA, vocab=VOCAB)
    m = Mamba(vocab=VOCAB + 2, d_model=20, d_state=8, n_layer=2, selective=False, seed=seed)
    fit(m, lambda n, r: mk(n, rng=r), steps=STEPS, batch=32, lr=3e-3, seed=seed)
    tok, exact = score(m.forward(selective_copy(1000, L, N_DATA, VOCAB, np.random.default_rng(777 + seed))[0]),
                       selective_copy(1000, L, N_DATA, VOCAB, np.random.default_rng(777 + seed))[1])
    lt, le = score(*(lambda b: (m.forward(b[0]), b[1]))(selective_copy(1000, 2 * L, N_DATA, VOCAB, np.random.default_rng(888 + seed))))
    r = dict(mode="lti-wide", seq_len=L, seed=seed, params=m.n_params(), tok=tok, exact=exact, long_tok=lt, long_exact=le)
    print(json.dumps(r), flush=True)
    with open(Path(__file__).with_name("param_matched.jsonl"), "a") as f:
        f.write(json.dumps(r) + "\n")
