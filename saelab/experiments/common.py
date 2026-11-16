import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from saelab import (SAE, coefficient_ratio, dead_latents, feature_recall, feature_recovery, fit,
                    fvu, l0, latent_purity, make_world)


def evaluate(sae, world, seed, n=8192):
    x, a = world.sample(n, np.random.default_rng(10_000 + seed))
    z = sae.encode(x)
    dec = sae.params["W_dec"]
    dead = dead_latents(z)
    r = feature_recovery(world.directions, dec)
    return {
        "L0": l0(z), "FVU": fvu(x, sae.decode(z)), "matched": r["matched"], "mean_cos": r["mean_best_cos"],
        "purity": latent_purity(world.directions, dec, alive=~dead), "dead": float(dead.sum()),
        "ratio": coefficient_ratio(a, z, world.directions, dec), "recall": feature_recall(a, z, world.directions, dec),
    }


def mean_std(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return np.nanmean(v), np.nanstd(v)


def fmt(rows, key, digits=3):
    m, s = mean_std(rows, key)
    return f"{m:.{digits}f}±{s:.{digits}f}"
