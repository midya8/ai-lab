"""Regenerate results/*.json (about a minute on 2 CPU cores): python make_results.py"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import gplab.studies as S  # noqa: E402

JOBS = {"hyper": S.hyperparameter_recovery, "sparse": S.sparse_methods, "features": S.feature_approximations,
        "calibration": S.calibration, "extrapolation": S.extrapolation, "scaling": S.scaling}

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    for name, fn in JOBS.items():
        t = time.time()
        with open(f"results/{name}.json", "w") as f:
            json.dump(fn(), f, indent=1)
        print(f"{name}: {time.time() - t:.1f}s")
