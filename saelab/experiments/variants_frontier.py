"""Study 5: JumpReLU and Gated SAEs vs BatchTopK / TopK / L1-ReLU on the Study-1 world
(64 true features in d=32, true L0 = 3, 128 latents, 5 seeds)."""
from common import *
from saelab import GatedSAE, JumpReLUSAE

SEEDS = range(5)
CONFIGS = ([("jumprelu", {"l0_coef": v}) for v in (0.01, 0.02, 0.03, 0.05, 0.1)]
           + [("gated", {"l1": v}) for v in (0.2, 0.4, 0.8, 1.6)]
           + [("relu", {"l1": 0.4})] + [("topk", {"k": 3})] + [("batchtopk", {"k": 3}), ("batchtopk", {"k": 4})])


def build(mode, kw, s):
    if mode == "jumprelu":
        return JumpReLUSAE(32, 128, seed=s, **kw)
    if mode == "gated":
        return GatedSAE(32, 128, seed=s, **kw)
    return SAE(32, 128, mode=mode, seed=s, **kw)


if __name__ == "__main__":
    print("d=32, 64 true features (true L0=3), 128 latents, 3000 steps x batch 512, 5 seeds; mean±std")
    print(f"{'config':<22}{'L0':>12}{'FVU':>14}{'matched':>14}{'purity':>14}{'dead':>14}{'ratio':>14}{'recall':>14}")
    for mode, kw in CONFIGS:
        rows = []
        for s in SEEDS:
            w = make_world(64, 32, l0=3, seed=s)
            sae = build(mode, kw, s)
            fit(sae, w, steps=3000, seed=s)
            rows.append(evaluate(sae, w, s))
        name = f"{mode} " + ",".join(f"{k}={v}" for k, v in kw.items())
        print(f"{name:<22}" + "".join(f"{fmt(rows, k, 2 if k in ('L0', 'dead') else 3):>14}" for k in ("L0", "FVU", "matched", "purity", "dead", "ratio", "recall")), flush=True)
