"""Standard vs FGSM-adversarially-trained CNN, 3 seeds.

Trains on 4000 synthetic images, tests on a disjoint 800-image set. Attack
accuracy is measured on the first 400 test images. Run:
    PYTHONPATH=src python experiments/adv_training.py
"""
import numpy as np

from visionlab.data import make_dataset
from visionlab.model import ModelConfig, build_cnn
from visionlab.robust import attack_report, natural_report
from visionlab.train import train

EPS = (0.0, 0.02, 0.05, 0.1)
SEEDS = (0, 1, 2)
Xte, yte = make_dataset(800, seed=999)


def run(adv_eps, seed):
    Xtr, ytr = make_dataset(4000, seed=seed * 10 + 1)
    m = build_cnn(ModelConfig(seed=seed))
    train(m, Xtr, ytr, epochs=12, seed=seed, adv_eps=adv_eps, log=None)
    acc = float((m.predict(Xte) == yte).mean())
    att = attack_report(m, Xte[:400], yte[:400], eps=EPS)
    nat = natural_report(m, Xte, yte, noise=(0.3,), shifts=(2,), contrasts=(0.5,))
    return acc, att, nat


rows = {}
for name, e in [("standard", 0.0), ("adv-trained (eps=0.05)", 0.05)]:
    res = [run(e, s) for s in SEEDS]
    rows[name] = res
    print(f"\n{name}")
    print(f"  clean acc          {np.mean([r[0] for r in res]):.3f} +- {np.std([r[0] for r in res]):.3f}")
    for atk in ("fgsm", "pgd"):
        print(f"  {atk:<4} " + "  ".join(f"eps={e_}: {np.mean([r[1][atk][e_] for r in res]):.3f}" for e_ in EPS))
    print(f"  noise 0.3: {np.mean([r[2]['noise'][0.3] for r in res]):.3f}  shift 2px: {np.mean([r[2]['shift'][2] for r in res]):.3f}  contrast 0.5: {np.mean([r[2]['contrast'][0.5] for r in res]):.3f}")
