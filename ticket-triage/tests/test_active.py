import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from triage.data import generate
from triage.model import TicketClassifier
from triage.active import uncertainty, select_batch, review_queue, simulate, compare

P = np.array([[0.98, 0.01, 0.01], [0.5, 0.49, 0.01], [0.34, 0.33, 0.33]])


@pytest.mark.parametrize("s", ["least_confidence", "margin", "entropy"])
def test_uncertainty_ranks_flat_over_confident(s):
    u = uncertainty(P, s)
    assert u[0] < u[1] and u[0] < u[2]


def test_margin_flags_near_tie_most():
    assert np.argmax(uncertainty(P, "margin")) == 1 or np.argmax(uncertainty(P, "margin")) == 2
    assert uncertainty(P, "margin")[1] > 0.9


def test_entropy_max_for_uniform():
    assert uncertainty(np.full((1, 4), 0.25), "entropy")[0] == pytest.approx(np.log(4))


def test_unknown_strategy():
    with pytest.raises(ValueError):
        uncertainty(P, "nope")


@pytest.fixture(scope="module")
def model_and_pool():
    d = generate(200, seed=1)
    m = TicketClassifier().fit([x["text"] for x in d[:40]], [x["category"] for x in d[:40]])
    return m, [x["text"] for x in d[40:]]


def test_select_batch_size_and_unique(model_and_pool):
    m, texts = model_and_pool
    idx = select_batch(m, texts, 5, "entropy")
    assert len(idx) == 5 and len(set(idx)) == 5
    assert len(select_batch(m, texts[:3], 10, "margin")) == 3


def test_review_queue_sorted_by_score(model_and_pool):
    m, texts = model_and_pool
    q = review_queue(m, texts, k=6)
    scores = [r["score"] for r in q]
    assert scores == sorted(scores, reverse=True)
    assert q[0]["predicted"] in m.classes


def test_simulate_curve_shape_and_growth():
    pool, test = generate(150, seed=2), generate(100, seed=3)
    c = simulate(pool, test, "margin", seed_size=8, batch=4, rounds=3)
    assert [n for n, _ in c] == [8, 12, 16, 20]
    assert all(0 <= a <= 1 for _, a in c)


def test_simulate_deterministic():
    pool, test = generate(100, seed=2), generate(60, seed=3)
    assert simulate(pool, test, "entropy", rounds=2) == simulate(pool, test, "entropy", rounds=2)


def test_compare_returns_all_strategies():
    pool, test = generate(100, seed=2), generate(60, seed=3)
    r = compare(pool, test, seeds=(0,), rounds=2)
    assert set(r) == {"least_confidence", "margin", "entropy", "random"}
