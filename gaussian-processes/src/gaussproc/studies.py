"""Studies that probe where each GP approximation works and where it breaks.
Everything is synthetic with known generating processes; each study returns a JSON-able dict."""
from __future__ import annotations

import time

import numpy as np

from .data import sample_gp_data, seasonal_trend, smooth_function
from .features import FeatureGP, NystromFeatures, RandomFourierFeatures, kernel_approx_error
from .gp import ExactGP, coverage, gaussian_nlpd
from .kernels import RBF, Linear, Matern32, Matern52, Periodic
from .sparse import SparseGP, greedy_inducing, kmeans_inducing


def _stats(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}


# 1 ------------------------------------------------------------------------------------
def hyperparameter_recovery(ns=(20, 50, 100, 300), seeds=8, restarts=6, ell=0.7, var=1.5, noise=0.1):
    """Fit RBF hyper-parameters by marginal likelihood on data drawn from the model itself.
    Reports the spread of the recovered values and how often random restarts end in the best optimum."""
    rows = []
    for n in ns:
        L, V, N, share, flat = [], [], [], [], 0
        for s in range(seeds):
            X, y, _ = sample_gp_data(RBF(ell, var), n, noise, seed=100 * n + s)
            gp = ExactGP(RBF(1.0, 1.0), 0.5).fit(X, y, restarts=restarts, seed=s, init_scale=2.0)
            ll = np.array(gp.fit_log_)
            share.append(float(np.mean(ll > ll.max() - 0.01)))
            L.append(float(gp.kernel.lengthscale[0])); V.append(gp.kernel.variance); N.append(gp.noise)
            flat += int(gp.kernel.lengthscale[0] > 3 * ell)
        rows.append({"n": n, "lengthscale": _stats(L), "variance": _stats(V), "noise": _stats(N),
                     "restarts_in_best_basin": float(np.mean(share)), "seeds_with_flat_optimum": flat})
    return {"truth": {"lengthscale": ell, "variance": var, "noise": noise}, "seeds": seeds,
            "restarts": restarts, "rows": rows}


# 2 ------------------------------------------------------------------------------------
def sparse_methods(n=1500, ms=(4, 8, 12, 16, 24, 32), noise_var=0.09, seed=0):
    """VFE / FITC / DTC vs the exact GP as the number of inducing points grows."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-4, 4, (n, 1))
    y = smooth_function(X) + np.sqrt(noise_var) * rng.standard_normal(n)
    Xt = np.linspace(-4, 4, 500)[:, None]
    ft = smooth_function(Xt)
    yt = ft + np.sqrt(noise_var) * rng.standard_normal(500)

    ex = ExactGP(RBF(1.0, 1.0), 0.3).fit(X, y, max_iter=60)
    m, v = ex.predict(Xt, include_noise=True)
    exact = {"lml": ex.log_marginal_likelihood(), "noise": ex.noise, "rmse_f": float(np.sqrt(np.mean((m - ft) ** 2))),
             "nlpd": gaussian_nlpd(yt, m, v)}
    rows = []
    for M in ms:
        for method in ("vfe", "fitc", "dtc"):
            k = RBF(1.0, 1.0)
            Z = X[greedy_inducing(k, X, M, seed=seed)]       # chosen with the initial kernel (lengthscale 1)
            sp = SparseGP(k, Z, 0.3, method).fit(X, y)
            m, v = sp.predict(Xt, include_noise=True)
            vfe_bound = SparseGP(sp.kernel.copy(), Z, sp.noise, "vfe").set_data(X, y).bound()
            rows.append({"M": M, "M_used": int(Z.shape[0]), "method": method, "objective": sp.bound(), "vfe_bound_at_these_params": vfe_bound,
                         "noise": sp.noise, "lengthscale": float(sp.kernel.lengthscale[0]),
                         "rmse_f": float(np.sqrt(np.mean((m - ft) ** 2))), "nlpd": gaussian_nlpd(yt, m, v)})
    # inducing placement at fixed M (hypers fixed to the exact fit so only placement differs)
    place = []
    M = 12
    for name in ("greedy", "random", "kmeans"):
        vals = []
        for s in range(5):
            if name == "greedy":
                Z = X[greedy_inducing(ex.kernel, X, M, seed=s)]
            elif name == "random":
                Z = X[np.random.default_rng(s).choice(n, M, replace=False)]
            else:
                Z = kmeans_inducing(X, M, seed=s)
            sp = SparseGP(ex.kernel.copy(), Z, ex.noise, "vfe").set_data(X, y)
            vals.append((sp.bound(), sp.trace_gap()))
        place.append({"placement": name, "bound_mean": float(np.mean([v[0] for v in vals])),
                      "trace_gap_mean": float(np.mean([v[1] for v in vals]))})
    return {"n": n, "true_noise": noise_var, "exact": exact, "rows": rows, "placement_M12": place}


# 3 ------------------------------------------------------------------------------------
def feature_approximations(n=400, dim=2, ds=(10, 25, 50, 100, 200, 500), seeds=5, noise=0.1):
    """Kernel error and predictive gap to the exact GP at equal feature budget (hypers fixed at truth)."""
    rows = []
    for D in ds:
        acc: dict[str, list] = {k: [] for k in ("rff", "orf", "nystrom_random", "nystrom_greedy")}
        for s in range(seeds):
            kern = RBF(1.0, 1.0)
            X, y, _ = sample_gp_data(kern, n + 300, noise, seed=s, dim=dim, low=-2, high=2)
            Xtr, ytr, Xte = X[:n], y[:n], X[n:]
            ex = ExactGP(kern.copy(), noise).set_data(Xtr, ytr)
            mex, vex = ex.predict(Xte)
            K = kern(Xtr)
            makers = {
                "rff": lambda: RandomFourierFeatures(kern, D, dim, seed=s),
                "orf": lambda: RandomFourierFeatures(kern, D, dim, seed=s, orthogonal=True),
                "nystrom_random": lambda: NystromFeatures(kern, Xtr[np.random.default_rng(s).choice(n, min(D, n), replace=False)]),
                "nystrom_greedy": lambda: NystromFeatures(kern, Xtr[greedy_inducing(kern, Xtr, min(D, n), seed=s)]),
            }
            for name, mk in makers.items():
                feat = mk()
                err = kernel_approx_error(K, feat(Xtr))
                fg = FeatureGP(feat, noise).fit(Xtr, ytr)
                m, v = fg.predict(Xte)
                acc[name].append((err, float(np.linalg.norm(m - mex) / np.linalg.norm(mex)), float(np.mean(v / vex))))
        row = {"D": D}
        for name, vals in acc.items():
            a = np.array(vals)
            row[name] = {"kernel_err": float(a[:, 0].mean()), "mean_rel_gap": float(a[:, 1].mean()), "var_ratio": float(a[:, 2].mean())}
        rows.append(row)
    return {"n": n, "dim": dim, "noise": noise, "rows": rows}


# 4 ------------------------------------------------------------------------------------
def calibration(n=80, seeds=20, noise=0.05):
    """Fit RBF / Matern-3/2 / Matern-5/2 to data from each true family; report coverage of
    95% intervals, NLPD, and whether the marginal likelihood or closed-form LOO picks the truth."""
    truths = {"rbf": RBF(0.9, 1.0), "matern32": Matern32(0.9, 1.0), "matern52": Matern52(0.9, 1.0)}
    models = {"rbf": RBF, "matern32": Matern32, "matern52": Matern52}
    rows = []
    for tname, tk in truths.items():
        agg = {m: {"cov": [], "nlpd": [], "lml": [], "loo": []} for m in models}
        for s in range(seeds):
            X, y, _ = sample_gp_data(tk, n + 400, noise, seed=7 * s + 1, low=-4, high=4)
            idx = np.random.default_rng(s).permutation(n + 400)
            tr, te = idx[:n], idx[n:]
            for mname, cls in models.items():
                gp = ExactGP(cls(1.0, 1.0), 0.2).fit(X[tr], y[tr], restarts=2, seed=s)
                m, v = gp.predict(X[te], include_noise=True)
                a = agg[mname]
                a["cov"].append(coverage(y[te], m, v)); a["nlpd"].append(gaussian_nlpd(y[te], m, v))
                a["lml"].append(gp.log_marginal_likelihood()); a["loo"].append(gp.loo_log_predictive())
        names = list(models)
        lml_pick = [names[i] for i in np.argmax([agg[m]["lml"] for m in names], axis=0)]
        loo_pick = [names[i] for i in np.argmax([agg[m]["loo"] for m in names], axis=0)]
        rows.append({"truth": tname,
                     "models": {m: {"coverage95": float(np.mean(a["cov"])), "nlpd": float(np.mean(a["nlpd"])),
                                    "lml": float(np.mean(a["lml"])), "loo": float(np.mean(a["loo"]))} for m, a in agg.items()},
                     "lml_picks_truth": float(np.mean([p == tname for p in lml_pick])),
                     "loo_picks_truth": float(np.mean([p == tname for p in loo_pick]))})
    return {"n_train": n, "seeds": seeds, "noise": noise, "rows": rows}


# 5 ------------------------------------------------------------------------------------
def extrapolation(n=160, seed=0, noise=0.1):
    """Trend + seasonality on t in [0, 12]; score interpolation (t<12) and extrapolation (12 to 16).
    Structured kernels are fit with a grid over initial periods and keep the best marginal likelihood."""
    t, y, f = seasonal_trend(n, noise, seed=seed)
    # the noise-free truth on a grid reaching past the training range (amplitude drift period fixed at 2.5 * 12)
    tg = np.linspace(0, 16, 400)
    amp = 1.0 + 0.3 * np.sin(2 * np.pi * tg / (2.5 * 12.0))
    fg = 0.35 * tg + amp * np.sin(2 * np.pi * tg)
    Tg = tg[:, None]
    inside, outside = tg < 12, tg >= 12

    def fit_best(make, periods=(None,)):
        best = None
        for p in periods:
            gp = ExactGP(make(p), 0.2, center=True)
            try:
                gp.fit(t, y, restarts=1, seed=1)
            except np.linalg.LinAlgError:
                continue
            if best is None or gp.log_marginal_likelihood() > best.log_marginal_likelihood():
                best = gp
        return best

    periods = (0.4, 0.7, 1.0, 1.6, 2.5, 4.0)
    models = {
        "rbf": (lambda p: RBF(1.0, 1.0), (None,)),
        "matern32+linear": (lambda p: Matern32(2.0, 1.0) + Linear(0.1), (None,)),
        "periodic+linear": (lambda p: Periodic(1.0, p, 1.0) + Linear(0.1), periods),
        "periodic*rbf+linear": (lambda p: Periodic(1.0, p, 1.0) * RBF(8.0, 1.0) + Linear(0.1), periods),
    }
    rows = []
    for name, (mk, per) in models.items():
        gp = fit_best(mk, per)
        m, v = gp.predict(Tg)
        row = {"model": name, "lml": gp.log_marginal_likelihood(), "n_params": int(gp.params.size),
               "rmse_interp": float(np.sqrt(np.mean((m - fg)[inside] ** 2))),
               "rmse_extrap": float(np.sqrt(np.mean((m - fg)[outside] ** 2))),
               "extrap_cov95_f": coverage(fg[outside], m[outside], v[outside]),
               "noise": gp.noise}
        if "periodic" in name:
            k = gp.kernel.a            # Sum(a, Linear); a is Periodic or Periodic * RBF
            row["period"] = float(k.a.period if hasattr(k, "a") else k.period)
        rows.append(row)
    return {"n": n, "noise": noise, "true_period": 1.0, "rows": rows}


# 6 ------------------------------------------------------------------------------------
def scaling(ns=(250, 500, 1000, 2000), M=50, D=200, dim=2, reps=3):
    """Wall-clock and memory of one objective evaluation + prediction at fixed hypers."""
    rows = []
    for n in ns:
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, (n, dim))
        y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)
        Xt = rng.uniform(-3, 3, (500, dim))
        kern = RBF(1.0, 1.0)

        def timeit(fn):
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
            return float(np.median(ts))

        def exact():
            g = ExactGP(kern.copy(), 0.1).set_data(X, y); g.lml_and_grad(); g.predict(Xt)

        Z = X[greedy_inducing(kern, X, M)]

        def vfe():
            s = SparseGP(kern.copy(), Z, 0.1, "vfe").set_data(X, y); s.bound(); s.predict(Xt)

        def rff():
            FeatureGP(RandomFourierFeatures(kern, D, dim), 0.1).fit(X, y).predict(Xt)

        rows.append({"n": n, "exact_s": timeit(exact), "vfe_s": timeit(vfe), "rff_s": timeit(rff),
                     "exact_bytes": 8 * n * n, "vfe_bytes": 8 * n * M, "rff_bytes": 8 * n * D})
    return {"M": M, "D": D, "dim": dim, "rows": rows}
