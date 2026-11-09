"""Uniform vs border-concentrated (instance-dependent) noise, same 20% rate, 3 seeds."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from datadoctor.synth import make_blobs, inject_label_noise, inject_instance_noise
from datadoctor.cv import cross_val_probs
from datadoctor.models import SoftmaxRegression, KNNClassifier
from datadoctor.confident import find_label_issues, confident_joint, calibrate_joint
from datadoctor.evaluate import mask_scores

print("| noise | model | true rate | est. rate | precision | recall | F1 |")
print("|---|---|---|---|---|---|---|")
for nname, inj in [("symmetric", lambda X, y, s: inject_label_noise(y, 0.2, seed=s)),
                   ("instance", lambda X, y, s: inject_instance_noise(X, y, 0.2, seed=s))]:
    for mname, fac in [("softmax", lambda: SoftmaxRegression()), ("kNN 15", lambda: KNNClassifier(15))]:
        R = []
        for s in range(3):
            X, y = make_blobs(seed=s)
            yn, fl = inj(X, y, s)
            K = int(yn.max()) + 1
            P = cross_val_probs(X, yn, fac, k=5, seed=s, n_classes=K)
            C, _ = confident_joint(P, yn); Cc = calibrate_joint(C, yn)
            sc = mask_scores(find_label_issues(P, yn), fl)
            R.append([fl.mean(), 1 - np.trace(Cc) / Cc.sum(), sc["precision"], sc["recall"], sc["f1"]])
        print(f"| {nname} | {mname} | " + " | ".join(f"{v:.3f}" for v in np.mean(R, 0)) + " |")
