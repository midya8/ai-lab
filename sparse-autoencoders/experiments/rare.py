"""Study 4: do rare features survive? Zipf-skewed firing probabilities (rank^-1), 64 features,
d=32, true L0=3. Report recovery of the 16 most frequent vs the 16 rarest features."""
from common import *
from sparse_ae.metrics import cosine_matrix

SEEDS = range(5)
print("d=32, 64 features, zipf=1, true L0=3, 128 latents; fraction of features with best cos>=0.9 (mean±std, 5 seeds)")
print(f"{'config':<18}{'FVU':>14}{'freq top16':>14}{'freq bottom16':>16}{'max p (bot16)':>15}", flush=True)
for mode, kw in [("relu", {"l1": 0.1}), ("relu", {"l1": 0.4}), ("topk", {"k": 3}), ("batchtopk", {"k": 3})]:
    rows = []
    for s in SEEDS:
        w = make_world(64, 32, l0=3, zipf=1.0, seed=s)
        sae = SAE(32, 128, mode=mode, seed=s, **kw)
        fit(sae, w, steps=3000, seed=s)
        r = evaluate(sae, w, s)
        best = cosine_matrix(w.directions, sae.params["W_dec"]).max(axis=1)
        r["top"] = float((best[:16] >= 0.9).mean())
        r["bot"] = float((best[-16:] >= 0.9).mean())
        r["pbot"] = float(w.probs[-16])
        rows.append(r)
    name = f"{mode} " + ",".join(f"{k}={v}" for k, v in kw.items())
    print(f"{name:<18}{fmt(rows, 'FVU'):>14}{fmt(rows, 'top'):>14}{fmt(rows, 'bot'):>16}{mean_std(rows, 'pbot')[0]:>15.4f}", flush=True)
