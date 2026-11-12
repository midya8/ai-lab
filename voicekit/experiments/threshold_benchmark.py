"""Per-word calibrated thresholds vs a single global threshold on streaming word spotting.

Run: PYTHONPATH=src python experiments/threshold_benchmark.py
Train seed 0, validation seed 3100 (disjoint), test streams seeds 9500+. Unseen speakers/noise in test.
Global thresholds swept over a grid; the best global one is chosen ON VALIDATION STREAMS, not test.
"""
import json
import numpy as np
from voicekit import CLASSES, MLP, Recognizer, build_dataset, render_stream, score_events
from voicekit.pipeline import WORDS, calibrate_thresholds
from voicekit.synth import OOV_TEST

X, y, _, _ = build_dataset(150, 0)
mlp = MLP(seed=0).fit(X, y, n_classes=len(CLASSES))
Xv, yv, _, _ = build_dataset(30, 3100, (5, 20), oov=OOV_TEST)
per_word = calibrate_thresholds(mlp, Xv, yv)
print("per-word thresholds:", {k: round(v, 2) for k, v in per_word.items()})


def run(thr, seed0, snr, n=12):
    agg = {"tp": 0, "fp": 0, "fn": 0, "wrong_word": 0}
    rng = np.random.default_rng(seed0 + snr)
    for k in range(n):
        words = [WORDS[i] for i in rng.integers(0, len(WORDS), 8)]
        sig, truth = render_stream(words, rng, snr, noise_kind=("white", "pink", "hum")[k % 3])
        s = score_events(Recognizer(mlp, threshold=thr).process(sig), truth)
        for key in agg:
            agg[key] += s[key]
    return agg


def f1(a):
    p = a["tp"] / max(a["tp"] + a["fp"], 1); r = a["tp"] / max(a["tp"] + a["fn"], 1)
    return 2 * p * r / (p + r) if p + r else 0.0


def pooled(thr, seed0, snrs=(20, 10, 5)):
    tot = {"tp": 0, "fp": 0, "fn": 0, "wrong_word": 0}
    for s in snrs:
        for k, v in run(thr, seed0, s).items():
            tot[k] += v
    return tot


grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
val = {g: f1(pooled(g, 4100)) for g in grid}
best_g = max(val, key=val.get)
print("validation F1 by global threshold:", {g: round(v, 3) for g, v in val.items()}, "-> best", best_g)
res = {"per_word_thresholds": per_word, "val_f1_global": val, "chosen_global": best_g, "test": {}}
for name, thr in (("global_0.6_default", 0.6), (f"global_{best_g}_val_best", best_g), ("per_word_calibrated", per_word)):
    a = pooled(thr, 9500)
    a["f1"] = f1(a)
    res["test"][name] = a
    print(name, a)
json.dump(res, open("experiments/threshold_results.json", "w"), indent=2)
