"""Does temperature scaling change confident-learning results? Small honest check (3 seeds)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from datadoctor.synth import make_blobs, inject_label_noise
from datadoctor.cv import cross_val_probs
from datadoctor.models import SoftmaxRegression, KNNClassifier
from datadoctor.calibration import calibrate, expected_calibration_error
from datadoctor.confident import find_label_issues, confident_joint, calibrate_joint
from datadoctor.evaluate import mask_scores

print("| model | seed | T | ECE raw->cal | est. noise raw | est. noise cal | true | F1 raw | F1 cal |")
print("|---|---|---|---|---|---|---|---|---|")
for name, fac in [("softmax l2=1e-2", lambda: SoftmaxRegression()), ("softmax l2=5", lambda: SoftmaxRegression(l2=5.0)), ("kNN k=15", lambda: KNNClassifier(15))]:
    for seed in range(3):
        X, y = make_blobs(seed=seed)
        yn, flipped = inject_label_noise(y, 0.2, seed=seed)[:2]
        K = int(yn.max()) + 1
        P = cross_val_probs(X, yn, fac, k=5, seed=seed, n_classes=K)
        Pc, T = calibrate(P, yn)
        def rate(Q):
            C, _ = confident_joint(Q, yn); Cc = calibrate_joint(C, yn)
            return 1 - np.trace(Cc) / Cc.sum()
        f = lambda Q: mask_scores(find_label_issues(Q, yn), flipped)["f1"]
        print(f"| {name} | {seed} | {T:.2f} | {expected_calibration_error(P, yn):.3f}->{expected_calibration_error(Pc, yn):.3f} | {rate(P):.3f} | {rate(Pc):.3f} | {flipped.mean():.3f} | {f(P):.3f} | {f(Pc):.3f} |")
