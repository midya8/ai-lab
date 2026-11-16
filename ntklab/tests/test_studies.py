import numpy as np
from ntklab import studies as S


def test_circle_points_unit_norm_and_jitter_sorted():
    X, phi = S.circle_points(10, jitter=0.6, seed=2)
    assert np.allclose((X ** 2).sum(1), 1.0)
    assert np.all(np.diff(phi) > 0)


def test_kernel_convergence_error_shrinks():
    r = S.kernel_convergence(widths=(16, 1024), seeds=4, n=10)
    assert r["rows"][1]["mean_rel_err"] < 0.5 * r["rows"][0]["mean_rel_err"]
    assert r["slope"] < -0.25


def test_lazy_vs_width_gap_shrinks():
    r = S.lazy_vs_width(widths=(8, 512), n_train=10, n_test=20, steps=200, seeds=2)
    a, b = r["rows"]
    assert b["func_gap"] < a["func_gap"]
    assert b["kernel_move"] < a["kernel_move"]


def test_spectral_bias_low_frequency_fits_first():
    r = S.spectral_bias(freqs=(1, 8), width=256, n=32, steps=(30,))
    lo, hi = r["rows"]
    assert lo["kernel_eig"] > hi["kernel_eig"]
    assert lo["measured"][30] < hi["measured"][30]
    assert lo["analytic"][30] < hi["analytic"][30]
    assert lo["linearised_same_init"][30] < hi["linearised_same_init"][30]


def test_depth_spectrum_runs_and_correlation_rises_with_depth():
    rows = S.depth_spectrum(depths=(1, 6), n=16)["rows"]
    assert rows[1]["mean_off_diag_corr"] > rows[0]["mean_off_diag_corr"]


def test_ensemble_mean_beats_single_net():
    r = S.ensemble_vs_analytic(width=128, members=8, n_train=8, n_test=16, steps=100)
    assert r["mean_vs_analytic_rel_err"] < r["single_net_vs_analytic_rel_err"]
