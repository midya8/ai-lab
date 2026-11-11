"""Does contrast augmentation fix the GAP head's contrast brittleness?

GAP + BN width 16, 3 seeds, 12 epochs; trained with photometric in {0, 0.5, 0.7}. Test on contrast x c
(the family it was trained on) and on contrast about the mean (x-0.5)*c+0.5 (not trained on). Run:
    PYTHONPATH=src python experiments/photometric.py
"""
import sys
import numpy as np
from visionlab.data import make_dataset
from visionlab.metrics import accuracy
from visionlab.model import ModelConfig, build_cnn
from visionlab.train import train

Xte, yte = make_dataset(1000, seed=999)
CS = (1.0, 0.5, 0.3)
print("| photometric | clean | x0.5 | x0.3 | about-mean 0.5 | about-mean 0.3 |\n|---|---|---|---|---|---|")
for ph in (0.0, 0.5, 0.7):
    rows = []
    for s in (0, 1, 2):
        Xtr, ytr = make_dataset(4000, seed=s * 10 + 1)
        m = build_cnn(ModelConfig(seed=s, head="gap", batchnorm=True, width=16))
        train(m, Xtr, ytr, epochs=12, seed=s, photometric=ph, log=None)
        r = [accuracy(yte, m.predict(np.clip(Xte * c, 0, 1).astype(Xte.dtype))) for c in CS[:3]]
        r += [accuracy(yte, m.predict(np.clip((Xte - .5) * c + .5, 0, 1).astype(Xte.dtype))) for c in (0.5, 0.3)]
        rows.append(r); print(f"  ph {ph} seed {s}: {r}", file=sys.stderr, flush=True)
    print(f"| {ph} | " + " | ".join(f"{v:.3f}" for v in np.mean(rows, 0)) + " |")
