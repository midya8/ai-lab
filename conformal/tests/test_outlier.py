import numpy as np
from conformal_pred import conformal_pvalues, benjamini_hochberg, conformal_outliers


def test_pvalue_formula():
    p = conformal_pvalues([1, 2, 3, 4], [0.5, 2.5, 10])
    assert np.allclose(p, [1.0, 3 / 5, 1 / 5])
    assert conformal_pvalues([1, 2, 2, 3], [2])[0] == (1 + 3) / 5   # ties count as >=


def test_pvalues_super_uniform():
    rng = np.random.default_rng(0)
    P = np.concatenate([conformal_pvalues(rng.normal(size=49), rng.normal(size=20)) for _ in range(3000)])
    for a in (0.05, 0.1, 0.5):
        assert abs(np.mean(P <= a) - a) < 0.012      # continuous scores: ~exactly uniform on grid


def test_bh_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = rng.uniform(size=15) ** 2
        got = benjamini_hochberg(p, 0.2)
        ps = np.sort(p)
        ks = [k for k in range(1, 16) if ps[k - 1] <= 0.2 * k / 15]
        want = p <= ps[max(ks) - 1] if ks else np.zeros(15, bool)
        assert (got == want).all()


def test_fdr_controlled_and_power():
    rng = np.random.default_rng(2)
    q, fdps, pw = 0.1, [], []
    for _ in range(400):
        cal = rng.normal(size=200)
        t = np.concatenate([rng.normal(size=160), rng.normal(4, 1, size=40)])
        flag, _ = conformal_outliers(cal, t, q)
        nf = flag.sum()
        fdps.append(flag[:160].sum() / max(nf, 1))
        pw.append(flag[160:].mean())
    assert np.mean(fdps) <= q * 160 / 200 + 0.015
    assert np.mean(pw) > 0.8
