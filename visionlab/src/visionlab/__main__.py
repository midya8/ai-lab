"""python -m visionlab {train,eval,explain,robust,show}"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import explain as ex
from . import png, robust
from .data import CLASSES, ascii_art, make_dataset
from .metrics import accuracy, confusion_matrix, format_report, wilson
from .model import ModelConfig, Sequential, build_cnn
from .train import train


def cmd_train(a):
    Xtr, ytr = make_dataset(a.train, seed=a.seed * 10 + 1, noise=a.noise)
    Xva, yva = make_dataset(a.val, seed=a.seed * 10 + 2, noise=a.noise)
    model = build_cnn(ModelConfig(width=a.width, hidden=a.hidden, dropout=a.dropout, seed=a.seed))
    print(f"{model.n_params():,} parameters, {a.train} train / {a.val} val images")
    train(model, Xtr, ytr, Xva, yva, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed,
          adv_eps=a.adv_eps, label_smoothing=a.label_smoothing)
    model.save(a.out)
    print(f"saved {a.out}")


def cmd_eval(a):
    model = Sequential.load(a.model)
    X, y = make_dataset(a.test, seed=a.seed, noise=a.noise)
    pred = model.predict(X)
    lo, hi = wilson(int((pred == y).sum()), len(y))
    print(f"test accuracy {accuracy(y, pred):.3f}  (95% Wilson CI {lo:.3f}-{hi:.3f}, n={len(y)})\n")
    print(format_report(confusion_matrix(y, pred, len(CLASSES)), CLASSES))


def cmd_robust(a):
    model = Sequential.load(a.model)
    X, y = make_dataset(a.test, seed=a.seed, noise=a.noise)
    print(robust.format_table(robust.natural_report(model, X, y), "natural perturbations"))
    print()
    print(robust.format_table(robust.attack_report(model, X[:a.attack_n], y[:a.attack_n]),
                              f"white-box attacks (L-inf eps, n={min(a.attack_n, len(y))})"))


def cmd_show(a):
    model = Sequential.load(a.model)
    X, y = make_dataset(a.n, seed=a.seed, noise=a.noise)
    P = model.predict_proba(X)
    for img, t, p in zip(X, y, P):
        print(ascii_art(img))
        print(f"true={CLASSES[t]} pred={CLASSES[p.argmax()]} ({p.max():.2f})\n")


def cmd_explain(a):
    model = Sequential.load(a.model)
    X, y = make_dataset(a.n, seed=a.seed, noise=a.noise)
    k = a.scale
    rows = []
    for x in X:
        cls = int(model.predict(x[None])[0])
        tiles = [png.gray_rgb(upscale_(x[0], k))]
        for fn in ex.METHODS.values():
            h = ex.normalize01(fn(model, x, cls))
            tiles.append(png.overlay(upscale_(x[0], k), upscale_(h, k)))
        rows.append(tiles)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    png.write_png(a.out, png.montage(rows))
    print(f"wrote {a.out}  columns: input | " + " | ".join(ex.METHODS))
    if a.faithfulness:
        Xe, ye = make_dataset(a.faithfulness, seed=a.seed + 1, noise=a.noise)
        rng = np.random.default_rng(0)
        print(f"\ndeletion AUC on {len(Xe)} correctly-classified images (lower = more faithful):")
        for name, fn in {**ex.METHODS, "random": None}.items():
            heat = (lambda x, c, fn=fn: fn(model, x, c)) if fn else (lambda x, c: ex.random_heat(x.shape[-2:], rng))
            score, _, n = ex.deletion_score(model, Xe, heat, only_correct=True, y=ye)
            print(f"  {name:<11} {score:.3f}  (n={n})")


def upscale_(img, k):
    return png.upscale(img, k)


def main(argv=None):
    p = argparse.ArgumentParser(prog="visionlab", description="CNN from scratch in NumPy")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--train", type=int, default=4000)
    t.add_argument("--val", type=int, default=600)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--width", type=int, default=8)
    t.add_argument("--hidden", type=int, default=64)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--adv-eps", type=float, default=0.0, help="FGSM adversarial training strength")
    t.add_argument("--label-smoothing", type=float, default=0.0)
    t.add_argument("--noise", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="model.npz")
    t.set_defaults(fn=cmd_train)

    for name, fn in [("eval", cmd_eval), ("robust", cmd_robust), ("show", cmd_show), ("explain", cmd_explain)]:
        s = sub.add_parser(name)
        s.add_argument("--model", default="model.npz")
        s.add_argument("--seed", type=int, default=999, help="test-set seed (differs from training seeds)")
        s.add_argument("--noise", type=float, default=0.2)
        if name in ("eval", "robust"):
            s.add_argument("--test", type=int, default=1000)
        if name == "robust":
            s.add_argument("--attack-n", type=int, default=400)
        if name == "show":
            s.add_argument("-n", type=int, default=3)
        if name == "explain":
            s.add_argument("-n", type=int, default=6)
            s.add_argument("--scale", type=int, default=6)
            s.add_argument("--out", default="docs/explain.png")
            s.add_argument("--faithfulness", type=int, default=0, help="N images for deletion-AUC comparison")
        s.set_defaults(fn=fn)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
