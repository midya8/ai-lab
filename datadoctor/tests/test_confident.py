import numpy as np
import pytest

from datadoctor.confident import (
    calibrate_joint,
    class_thresholds,
    confident_joint,
    estimate_noise_matrix,
    estimate_prior,
    find_label_issues,
    label_quality_scores,
    normalized_margin,
    rank_label_issues,
)

# 3 classes; rows 0-3 labelled 0, 4-7 labelled 1, 8-11 labelled 2. Row 3 (labelled 0) is clearly a 1;
# row 7 (labelled 1) is clearly a 2.
P = np.array(
    [
        [0.9, 0.05, 0.05],
        [0.8, 0.1, 0.1],
        [0.85, 0.1, 0.05],
        [0.1, 0.85, 0.05],  # labelled 0, really 1
        [0.1, 0.8, 0.1],
        [0.05, 0.9, 0.05],
        [0.1, 0.85, 0.05],
        [0.05, 0.1, 0.85],  # labelled 1, really 2
        [0.05, 0.1, 0.85],
        [0.1, 0.1, 0.8],
        [0.05, 0.05, 0.9],
        [0.1, 0.05, 0.85],
    ]
)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])


def test_thresholds_are_mean_self_confidence():
    t = class_thresholds(P, Y)
    assert t[0] == pytest.approx((0.9 + 0.8 + 0.85 + 0.1) / 4)
    assert t[2] == pytest.approx(np.mean(P[8:, 2]))


def test_confident_joint_hand_computed():
    C, assigned = confident_joint(P, Y)
    # row 3 -> true class 1 ; row 7 -> true class 2
    assert assigned[3] == 1 and assigned[7] == 2
    assert C[0, 1] == 1 and C[1, 2] == 1
    # row 9 (P=0.80 for its class, threshold 0.85) clears no threshold and is left out
    assert assigned[9] == -1
    assert C.sum() == len(Y) - 1
    assert np.trace(C) == len(Y) - 3


def test_calibration_preserves_label_counts_and_total():
    C, _ = confident_joint(P, Y)
    Cc = calibrate_joint(C, Y)
    assert Cc.sum() == pytest.approx(len(Y))
    assert Cc.sum(axis=1) == pytest.approx(np.bincount(Y), rel=1e-9)


def test_noise_matrix_columns_sum_to_one_and_prior_sums_to_one():
    C, _ = confident_joint(P, Y)
    Cc = calibrate_joint(C, Y)
    N = estimate_noise_matrix(Cc)
    assert N.sum(axis=0) == pytest.approx(np.ones(3))
    assert estimate_prior(Cc).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("method", ["argmax", "confident_joint", "prune_by_class", "prune_by_noise_rate", "both"])
def test_all_methods_flag_the_two_planted_errors(method):
    mask = find_label_issues(P, Y, method)
    assert mask[3] and mask[7]
    assert mask.sum() == 2


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        find_label_issues(P, Y, "nope")
    with pytest.raises(ValueError):
        label_quality_scores(P, Y, "nope")


def test_ranking_puts_planted_errors_first():
    for method in ("self_confidence", "margin", "entropy"):
        order = rank_label_issues(P, Y, method)
        assert set(order[:2]) == {3, 7}


def test_normalized_margin_sign():
    m = normalized_margin(P, Y)
    assert m[3] < 0 and m[7] < 0 and (np.delete(m, [3, 7]) > 0).all()


def test_perfect_labels_flag_nothing():
    Pp = np.eye(3)[np.repeat([0, 1, 2], 10)] * 0.9 + 0.05
    y = np.repeat([0, 1, 2], 10)
    for method in ("argmax", "confident_joint", "prune_by_class", "prune_by_noise_rate", "both"):
        assert find_label_issues(Pp, y, method).sum() == 0


def test_class_with_no_examples_does_not_crash():
    Pp = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 0, 1, 1])  # class 2 absent
    mask = find_label_issues(Pp, y, "both")
    assert mask.shape == (4,)
