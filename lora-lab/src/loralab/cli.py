import argparse
import numpy as np
from .lora import MLP, make_task


def run(rank=2, steps=300, seed=0, rot=0.8):
    rng = np.random.default_rng(seed)
    d, k = 32, 6
    xa, ya = make_task(2000, d, k, 1)
    xb, yb = make_task(400, d, k, 2, rot); xbt, ybt = make_task(1000, d, k, 3, rot)
    m = MLP(d, 64, k, rng)
    for _ in range(steps):
        m.step_full(xa, ya, 0.3)
    out = {"base_acc_task_A": float((m.predict(xa) == ya).mean()),
           "base_acc_task_B": float((m.predict(xbt) == ybt).mean())}
    m.attach_lora(rank)
    for _ in range(steps):
        m.step_lora(xb, yb, 0.3)
    out["lora_acc_task_B"] = float((m.predict(xbt) == ybt).mean())
    out["trainable_params"] = m.l1.n_trainable() + m.l2.n_trainable()
    out["full_params"] = m.W1.size + m.W2.size + 64 + k
    before = m.predict(xbt); m.merge()
    out["merge_preserves_predictions"] = bool((m.predict(xbt) == before).all())
    return out


def main():
    ap = argparse.ArgumentParser(description="LoRA adaptation demo")
    ap.add_argument("--rank", type=int, default=2)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    for k, v in run(a.rank, a.steps).items():
        print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    main()
