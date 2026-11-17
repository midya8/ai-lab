"""Study 6: feature absorption. 8 parents x 3 children (32 features) with child => parent, d=16
(2x superposition), 64 latents. Expected true L0 = 8*0.25 + 24*0.125 = 5. Two magnitude regimes:
'random' (U(0.5,1.5), parent and child magnitudes independent) and 'fixed' (all magnitudes 1.0, so
folding the parent into the child costs nothing)."""
import sys
from common import *
from sparse_ae import GatedSAE, JumpReLUSAE, absorption, make_hierarchical_world

SEEDS = range(5)
CONFIGS = [("relu", {"l1": 0.1}), ("relu", {"l1": 0.4}), ("topk", {"k": 5}), ("batchtopk", {"k": 5}),
           ("jumprelu", {"l0_coef": 0.03}), ("jumprelu", {"l0_coef": 0.1}), ("gated", {"l1": 0.4}), ("gated", {"l1": 1.6})]


def build(mode, kw, s, d=16, m=64):
    if mode == "jumprelu":
        return JumpReLUSAE(d, m, seed=s, **kw)
    if mode == "gated":
        return GatedSAE(d, m, seed=s, **kw)
    return SAE(d, m, mode=mode, seed=s, **kw)


def run(regime, seeds=SEEDS, configs=CONFIGS, child_cos=0.0, d=16):
    lo, hi = (0.5, 1.5) if regime == "random" else (1.0, 1.0)
    out = []
    for mode, kw in configs:
        rows = []
        for s in seeds:
            w = make_hierarchical_world(8, 3, d=d, child_cos=child_cos, seed=s, mag_lo=lo, mag_hi=hi)
            sae = build(mode, kw, s, d=d)
            fit(sae, w, steps=3000, seed=s)
            x, a = w.sample(8192, np.random.default_rng(10_000 + s))
            z = sae.encode(x)
            r = evaluate_world(sae, w, x, a, z)
            r.update(absorption(w, z, a, sae.params["W_dec"]))
            rows.append(r)
        out.append((mode, kw, rows))
    return out


def evaluate_world(sae, w, x, a, z):
    dec = sae.params["W_dec"]
    return {"L0": l0(z), "FVU": fvu(x, sae.decode(z)), "matched": feature_recovery(w.directions, dec)["matched"],
            "recall": feature_recall(a, z, w.directions, dec)}


if __name__ == "__main__":
    regime = sys.argv[1]
    cos = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    print(f"regime={regime} child_cos={cos}; 8 parents x 3 children, d=16, 64 latents, 5 seeds; mean±std")
    print(f"{'config':<22}{'L0':>12}{'FVU':>14}{'matched':>14}{'recall':>14}{'absorption':>14}{'p(no kid)':>14}{'p(kid)':>14}")
    for mode, kw, rows in run(regime, child_cos=cos):
        name = f"{mode} " + ",".join(f"{k}={v}" for k, v in kw.items())
        print(f"{name:<22}" + "".join(f"{fmt(rows, k, 2 if k == 'L0' else 3):>14}" for k in ("L0", "FVU", "matched", "recall", "absorption", "rate_no_child", "rate_with_child")), flush=True)
