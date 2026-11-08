"""Dense head vs global-average-pool head (each with/without BatchNorm) under translation.

The dense head wires every spatial position to its own weights, so moving a shape moves the
evidence to weights that never saw it. A conv + global-average-pool head sums evidence over
positions and is (nearly) translation invariant by construction. 5 architectures x 3 seeds,
12 epochs on 4000 synthetic images, tested on a disjoint 1000-image set. Run:
    PYTHONPATH=src python experiments/heads_shift.py
"""
import sys

import numpy as np

from visionlab.data import make_dataset
from visionlab.model import ModelConfig, build_cnn
from visionlab.robust import natural_report
from visionlab.train import train

SEEDS = (0, 1, 2)
SHIFTS = (0, 1, 2, 3, 4)
CONFIGS = {
    "dense": dict(head="dense", batchnorm=False),
    "dense + BN": dict(head="dense", batchnorm=True),
    "GAP": dict(head="gap", batchnorm=False),
    "GAP + BN": dict(head="gap", batchnorm=True),
    "GAP + BN, width 16": dict(head="gap", batchnorm=True, width=16),   # closer parameter budget
}
Xte, yte = make_dataset(1000, seed=999)

results = {}
for name, kw in CONFIGS.items():
    rows = []
    for s in SEEDS:
        Xtr, ytr = make_dataset(4000, seed=s * 10 + 1)
        m = build_cnn(ModelConfig(seed=s, **kw))
        train(m, Xtr, ytr, epochs=12, seed=s, log=None)
        rep = natural_report(m, Xte, yte, noise=(0.3,), shifts=SHIFTS, contrasts=(0.5,))
        rows.append((m.n_params(), rep))
        print(f"  {name} seed {s}: clean {rep['shift'][0]:.3f}", file=sys.stderr, flush=True)
    results[name] = rows

hdr = "| head | params | clean | " + " | ".join(f"shift {k}px" for k in SHIFTS[1:]) + " | noise 0.3 | contrast 0.5 |"
print(hdr)
print("|" + "---|" * (hdr.count("|") - 1))
for name, rows in results.items():
    mean = lambda f: np.mean([f(r) for _, r in rows])
    cells = [f"{mean(lambda r: r['shift'][k]):.3f}" for k in SHIFTS]
    sd = np.std([r["shift"][0] for _, r in rows])
    print(f"| {name} | {rows[0][0]:,} | {cells[0]} (±{sd:.3f}) | " + " | ".join(cells[1:]) +
          f" | {mean(lambda r: r['noise'][0.3]):.3f} | {mean(lambda r: r['contrast'][0.5]):.3f} |")
