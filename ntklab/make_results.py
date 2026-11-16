"""Regenerate results/*.json and results/summary.md (about a minute)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ntklab import studies as S

os.makedirs("results", exist_ok=True)
runs = {"convergence": S.kernel_convergence, "lazy": S.lazy_vs_width, "ensemble": S.ensemble_vs_analytic,
        "spectral": S.spectral_bias, "alpha": S.lazy_alpha, "depth": S.depth_spectrum}
out = {}
for name, fn in runs.items():
    t = time.time(); out[name] = fn()
    json.dump(out[name], open(f"results/{name}.json", "w"), indent=1)
    print(name, f"{time.time()-t:.1f}s", flush=True)
