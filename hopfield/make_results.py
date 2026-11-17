"""Regenerate results/*.md (and *.json) from the studies. Takes about 45 s on 2 cores."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from hopfield import studies as S  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

JOBS = {
    "capacity": lambda: S.capacity_study(),
    "stress_noise25": lambda: S.capacity_study(n=100, ps=(100, 400, 1000, 2000, 4000), noise=0.25, queries=20,
                                               rules=("dam-poly3", "dam-poly5", "dam-exp")),
    "basin": lambda: S.basin_study(),
    "correlated": lambda: S.correlated_study(),
    "beta_rho0": lambda: S.beta_study(),
    "beta_rho05": lambda: S.beta_study(rho=0.5),
    "glyphs": lambda: S.glyph_study(),
}

for name, fn in JOBS.items():
    rows = fn()
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(S.fmt_table(rows))
    print("wrote", name)
with open(os.path.join(OUT, "glyph_example.txt"), "w") as f:
    f.write(S.glyph_example("dam-exp", 3) + "\n")
