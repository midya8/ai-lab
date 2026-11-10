import numpy as np
from microgrid.parity import compare


def test_logits_loss_and_all_grads_match_tinygpt():
    r = compare()
    assert r["n_params"] == 6 + 12 * 2
    assert r["logits"] < 1e-9 and r["loss"] < 1e-9
    assert r["grad"] < 1e-8, r["worst_param"]


def test_parity_other_shapes():
    r = compare(vocab=7, d=12, heads=3, layers=1, ctx=5, batch=2, seed=4)
    assert r["logits"] < 1e-9 and r["grad"] < 1e-8
