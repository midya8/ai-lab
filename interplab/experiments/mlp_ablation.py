"""MLP ablation on the trained induction model (seeds 0-2): is the MLP needed for the copy rule?"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from interplab import train_induction, InterpModel, make_batch, mlp_ablation_table, neuron_ablation_table

for seed in (0, 1, 2):
    m, _ = train_induction(seed=seed)
    im = InterpModel.from_minigpt(m)
    tokens, periods = make_batch(np.random.default_rng(1000 + seed), 256)
    for mode in ("zero", "mean"):
        t = mlp_ablation_table(im, tokens, periods, mode)
        b = t["baseline"]
        print(f"seed {seed} {mode}-ablate whole MLP: base nll {b['nll']:.3f} acc {b['acc']:.3f} | " +
              " | ".join(f"L{r['layer']}: nll {r['nll']:.3f} acc {r['acc']:.3f}" for r in t["rows"]))
    for l in range(im.layers):
        n = neuron_ablation_table(im, tokens, periods, l, "mean", top=10)
        print(f"seed {seed} L{l} single-neuron mean-ablation: max d_nll {n['d_nll'].max():.4f}, "
              f"median {np.median(n['d_nll']):.5f}, top10 share {n['top_share']:.2f}, "
              f"top10 joint {n['joint_top_d_nll']:.4f} vs sum {n['sum_top_d_nll']:.4f}")
