"""Experiments that put numbers on the NTK picture.  Each returns plain dicts/lists."""
from __future__ import annotations

import numpy as np

from .analytic import ntk as ntk_inf
from .net import MLP
from .regression import gd_predict, residual_modes, stable_lr
from .train import kernel_change, train_gd


def circle_points(n: int, jitter: float = 0.0, seed: int = 0):
    """n points on the unit circle (x = (cos phi, sin phi)); returns (X, phi)."""
    phi = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if jitter:
        phi = phi + np.random.default_rng(seed).uniform(-jitter, jitter, n) * (2 * np.pi / n)
    return np.stack([np.cos(phi), np.sin(phi)], 1), phi


def target_fn(phi):
    return np.sin(phi) + 0.5 * np.sin(3 * phi + 0.4)


def _slope(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


# ---------------------------------------------------------------------------
def kernel_convergence(widths=(16, 64, 256, 1024, 4096), depth=2, n=24, seeds=8, sb2=0.1):
    """Relative error ||Theta_hat - Theta_inf||_F / ||Theta_inf||_F against width."""
    X, _ = circle_points(n)
    Kinf = ntk_inf(X, depth=depth, sb2=sb2)
    rows = []
    for w in widths:
        e = [np.linalg.norm(MLP(2, w, depth, sb2, seed=1000 * w + s).ntk(X) - Kinf) / np.linalg.norm(Kinf)
             for s in range(seeds)]
        rows.append({"width": w, "mean_rel_err": float(np.mean(e)), "std": float(np.std(e))})
    return {"depth": depth, "rows": rows,
            "slope": _slope([r["width"] for r in rows], [r["mean_rel_err"] for r in rows])}


# ---------------------------------------------------------------------------
def lazy_vs_width(widths=(8, 32, 128, 512, 2048), depth=2, n_train=16, n_test=64, steps=600,
                  seeds=6, sb2=0.1):
    """Train real nets with GD and compare with the linearised model at the *same* init.
    The step is 0.5 x the stability limit of each net's own initial kernel.

    gap      = ||f_trained - f_linearised||_2 / ||f_linearised||_2 on test points
    kernel   = ||Theta_T - Theta_0||_F / ||Theta_0||_F after training
    """
    X, phi = circle_points(n_train, jitter=0.6, seed=1)
    Xt, pt = circle_points(n_test)
    y = target_fn(phi)
    rows = []
    for w in widths:
        gap, kmove, loss, lin_loss = [], [], [], []
        for s in range(seeds):
            net = MLP(2, w, depth, sb2, seed=7000 + 31 * w + s)
            K, Kt = net.ntk(X), net.ntk(Xt, X)
            lr = stable_lr(K, 0.5)      # per-net step: narrow nets can have a larger lambda_max
            res = train_gd(net, X, y, lr, steps)
            tr_lin, te_lin = gd_predict(K, Kt, net(X), net(Xt), y, lr, steps)
            pred = res.predict(Xt)
            gap.append(np.linalg.norm(pred - te_lin) / np.linalg.norm(te_lin))
            kmove.append(kernel_change(res.net.ntk(X), K))
            loss.append(0.5 * np.mean((res.predict(X) - y) ** 2))
            lin_loss.append(0.5 * np.mean((tr_lin - y) ** 2))
        rows.append({"width": w, "func_gap": float(np.mean(gap)), "kernel_move": float(np.mean(kmove)),
                     "train_loss": float(np.mean(loss)), "linearised_train_loss": float(np.mean(lin_loss))})
    return {"depth": depth, "steps": steps, "rows": rows,
            "gap_slope": _slope([r["width"] for r in rows], [r["func_gap"] for r in rows]),
            "kernel_slope": _slope([r["width"] for r in rows], [r["kernel_move"] for r in rows])}


# ---------------------------------------------------------------------------
def ensemble_vs_analytic(width=512, depth=2, members=24, n_train=16, n_test=64, steps=300, sb2=0.1):
    """Mean over an ensemble of independently initialised, trained nets vs the analytic
    NTK-GP prediction with zero initial function at the same step count."""
    X, phi = circle_points(n_train, jitter=0.6, seed=1)
    Xt, _ = circle_points(n_test)
    y = target_fn(phi)
    K = ntk_inf(X, depth=depth, sb2=sb2)
    Kt = ntk_inf(Xt, X, depth=depth, sb2=sb2)
    lr = stable_lr(K, 0.5)
    z, zt = np.zeros(n_train), np.zeros(n_test)
    _, analytic = gd_predict(K, Kt, z, zt, y, lr, steps)
    preds = np.stack([train_gd(MLP(2, width, depth, sb2, seed=500 + m), X, y, lr, steps).predict(Xt)
                      for m in range(members)])
    mean = preds.mean(0)
    sem_rel = float(np.linalg.norm(preds.std(0, ddof=1) / np.sqrt(members)) / np.linalg.norm(analytic))
    return {"width": width, "members": members, "steps": steps,
            "mean_vs_analytic_rel_err": float(np.linalg.norm(mean - analytic) / np.linalg.norm(analytic)),
            "single_net_vs_analytic_rel_err": float(np.mean(
                [np.linalg.norm(p - analytic) / np.linalg.norm(analytic) for p in preds])),
            "expected_noise_floor": sem_rel}


# ---------------------------------------------------------------------------
def spectral_bias(freqs=(1, 2, 4, 8, 16), width=2048, depth=2, n=96, steps=(10, 50, 200, 1000), sb2=0.1):
    """Single-frequency targets sin(k phi): fraction ||f_t - y|| / ||y|| still unfit after t steps.

    Three columns per step count: the real wide net (measured), the linearised model at
    the *same* initialisation (its empirical kernel; isolates non-linearity) and the
    infinite-width analytic NTK prediction.  Also the kernel's Rayleigh quotient at that
    frequency (the speed at which the mode is learned)."""
    X, phi = circle_points(n)
    K = ntk_inf(X, depth=depth, sb2=sb2)
    lr = stable_lr(K, 0.5)
    net = MLP(2, width, depth, sb2, seed=11)
    Ke = net.ntk(X)
    z = np.zeros(n)
    steps = tuple(steps)
    rows = []
    for k in freqs:
        y = np.sin(k * phi)
        row = {"freq": k, "kernel_eig": float(y @ K @ y / (y @ y)),
               "measured": {}, "linearised_same_init": {}, "analytic": {}}

        def rec(t, work, y=y, row=row):
            if t in steps:
                out = work(X) - net(X)
                row["measured"][t] = float(np.linalg.norm(out - y) / np.linalg.norm(y))
        train_gd(net, X, y, lr, max(steps), center=True, record_every=1, on_record=rec)
        for t in steps:
            tr, _ = gd_predict(Ke, Ke, z, z, y, lr, t)
            row["linearised_same_init"][t] = float(np.linalg.norm(tr - y) / np.linalg.norm(y))
            tr, _ = gd_predict(K, K, z, z, y, lr, t)
            row["analytic"][t] = float(np.linalg.norm(tr - y) / np.linalg.norm(y))
        rows.append(row)
    return {"width": width, "n": n, "lr": lr, "steps": list(steps), "rows": rows}


# ---------------------------------------------------------------------------
def lazy_alpha(alphas=(1, 3, 10, 30, 100, 300), width=256, depth=2, n_train=16, n_test=64,
               steps=600, seeds=6, sb2=0.1):
    """Rich -> lazy: scale the output by alpha (centred at init) and shrink the step by
    alpha^2.  Function dynamics are unchanged but parameters (and the kernel) move ~1/alpha."""
    X, phi = circle_points(n_train, jitter=0.6, seed=1)
    Xt, pt = circle_points(n_test)
    y, yt = target_fn(phi), target_fn(pt)
    lr = stable_lr(ntk_inf(X, depth=depth, sb2=sb2), 0.5)
    rows = []
    for a in alphas:
        km, tl, te, pm = [], [], [], []
        for s in range(seeds):
            net = MLP(2, width, depth, sb2, seed=9000 + s)
            K0 = net.ntk(X)
            r = train_gd(net, X, y, lr, steps, alpha=float(a), center=True)
            km.append(kernel_change(r.net.ntk(X), K0))
            tl.append(0.5 * np.mean((r.predict(X) - y) ** 2))
            te.append(0.5 * np.mean((r.predict(Xt) - yt) ** 2))
            pm.append(np.linalg.norm(r.net.get_flat() - net.get_flat()))
        rows.append({"alpha": a, "kernel_move": float(np.median(km)), "param_move": float(np.mean(pm)),
                     "train_loss": float(np.mean(tl)), "test_loss": float(np.mean(te))})
    return {"width": width, "steps": steps, "rows": rows}


# ---------------------------------------------------------------------------
def depth_spectrum(depths=(1, 2, 3, 4, 6, 8), n=64, sb2=0.1):
    """Analytic NTK on a circle: how the spectrum (and so the optimisation speed) changes with depth."""
    X, _ = circle_points(n)
    rows = []
    for L in depths:
        K = ntk_inf(X, depth=L, sb2=sb2)
        lam = np.clip(np.linalg.eigvalsh(K)[::-1], 1e-300, None)
        p = lam / lam.sum()
        rows.append({"depth": L, "top_eig_over_n": float(lam[0] / n), "mean_off_diag_corr":
                     float((K / np.sqrt(np.outer(np.diag(K), np.diag(K))))[~np.eye(n, dtype=bool)].mean()),
                     "eff_rank": float(np.exp(-(p * np.log(p)).sum())),
                     "cond_top16": float(lam[0] / lam[15])})
    return {"rows": rows}
