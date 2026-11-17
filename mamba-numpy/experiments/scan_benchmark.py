"""Sequential vs parallel (associative) scan, and constant-state decoding vs a growing KV cache.

Run alone on an idle machine: wall-clock numbers are noisy otherwise. The scan is NumPy, so the
'parallel' version is vectorised over time but does O(T log T) work; on a GPU the depth, not the
work, is what matters. Both results are exact (max abs difference is printed).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mamba import Mamba, scan_parallel, scan_sequential  # noqa: E402


def best_of(fn, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)


rng = np.random.default_rng(0)
rows = []
print("scan over (B=4, T, D=32, N=8) float64, best of 5")
print(f"{'T':>6} {'sequential ms':>14} {'parallel ms':>12} {'max |diff|':>11}")
for T in [16, 64, 256, 1024, 2048]:
    a = rng.uniform(0.5, 1.0, (4, T, 32, 8))
    b = rng.normal(size=(4, T, 32, 8))
    ts, tp = best_of(lambda: scan_sequential(a, b)), best_of(lambda: scan_parallel(a, b))
    diff = float(np.abs(scan_sequential(a, b) - scan_parallel(a, b)).max())
    rows.append(dict(T=T, sequential_ms=1e3 * ts, parallel_ms=1e3 * tp, max_diff=diff))
    print(f"{T:>6} {1e3 * ts:>14.2f} {1e3 * tp:>12.2f} {diff:>11.1e}")

print("\nrecurrent decoding state vs a transformer KV cache (float64, same width/depth)")
m = Mamba(vocab=16, d_model=64, d_state=16, n_layer=4)
kv_per_token = 2 * 4 * 64 * 8  # K and V, 4 layers, d_model 64, 8 bytes
print(f"SSM state: {m.state_bytes():,} bytes per sequence, at every length")
for T in [128, 1024, 8192, 65536]:
    print(f"  KV cache at T={T:>6}: {kv_per_token * T:>12,} bytes  ({kv_per_token * T / m.state_bytes():.1f}x the SSM state)")

Path(__file__).with_name("scan_benchmark.json").write_text(json.dumps(rows, indent=1))
