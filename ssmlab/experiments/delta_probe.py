"""Train selective models on selective copying, then ask what delta does (3 seeds, L=32)."""
import functools, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ssmlab import Mamba, selective_copy
from ssmlab.train import fit
from ssmlab.analysis import delta_traces, gate_auc, skip_ablation

L, NDATA, V, STEPS = 32, 4, 4, 2500
out = []
for seed in [0, 1, 2]:
    mk = functools.partial(selective_copy, seq_len=L, n_data=NDATA, vocab=V)
    m = Mamba(vocab=V + 2, d_model=16, d_state=8, n_layer=2, selective=True, seed=seed)
    fit(m, lambda n, r: mk(n, rng=r), steps=STEPS, batch=32, lr=3e-3, seed=seed)
    toks, tg = selective_copy(500, L, NDATA, V, np.random.default_rng(777 + seed))
    g = gate_auc(delta_traces(m, toks), toks, L)
    a = skip_ablation(m, toks, tg, L, np.random.default_rng(5))
    out.append({"seed": seed, "gate": g, "ablation": a})
    print(seed, json.dumps(out[-1]), flush=True)
    Path("delta_probe.json").write_text(json.dumps(out, indent=1))
