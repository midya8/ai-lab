import numpy as np
import pytest
from interplab import (train_induction, InterpModel, make_batch, loss_and_acc, circuit_report, format_report)


@pytest.fixture(scope="module")
def trained():
    m, hist = train_induction(seed=0, steps=1200, log_every=400)
    return InterpModel.from_minigpt(m), hist


def test_training_learns_the_copy_rule(trained):
    im, hist = trained
    assert hist[-1][1] < hist[0][1]
    tokens, periods = make_batch(np.random.default_rng(77), 200)
    nll, acc = loss_and_acc(im, tokens, periods)
    chance = np.log(24)
    assert nll < 0.5 * chance and acc > 0.8


def test_report_formats(trained):
    im, _ = trained
    r = circuit_report(im, n=64)
    txt = format_report(r, im.layers, im.H)
    assert "L1H0" in txt and "direct logit attribution" in txt
    assert abs(sum(r["dla"].values()) - r["dla_total"]) < 1e-8


def test_trained_model_copies_from_the_true_source(trained):
    from interplab import copy_fidelity
    im, _ = trained
    tokens, periods = make_batch(np.random.default_rng(8), 150)
    r = copy_fidelity(im, tokens, periods, np.random.default_rng(1))
    assert r["follow"] > 0.6 and r["follow"] > r["control"] + 0.3


def test_mlp_ablation_tables(trained):
    from interplab import mlp_ablation_table, neuron_ablation_table
    im, _ = trained
    tokens, periods = make_batch(np.random.default_rng(5), 64)
    t = mlp_ablation_table(im, tokens, periods)
    assert len(t["rows"]) == im.layers
    n = neuron_ablation_table(im, tokens, periods, layer=0, top=5)
    assert n["d_nll"].shape[0] == 4 * im.D and n["d_nll"][n["order"][0]] >= n["d_nll"][n["order"][-1]]
    assert 0.0 <= n["top_share"] <= 1.0
