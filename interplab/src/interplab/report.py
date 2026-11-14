"""One-call circuit report for a trained model."""
import numpy as np
from .task import make_batch, corrupt_batch
from . import analysis as A


def circuit_report(model, n=256, seed=1234):
    rng = np.random.default_rng(seed)
    tokens, periods = make_batch(rng, n)
    _, cache = A._run(model, tokens[:, :-1])
    nll, acc = A.loss_and_acc(model, tokens, periods)
    prev = A.previous_token_scores(cache, model.layers)
    ind = A.induction_scores(cache, model.layers, periods)
    zero = A.head_ablation_table(model, tokens, periods, "zero")
    mean = A.head_ablation_table(model, tokens, periods, "mean")
    dla, total = A.logit_attribution(model, tokens, periods)
    corrupt = corrupt_batch(rng, tokens, periods)
    patch = A.patch_head_effects(model, tokens, corrupt, periods)
    k, k_ref = A.composition_scores(model, "k")
    copy = A.copy_fidelity(model, tokens, periods, np.random.default_rng(seed + 1))
    copy_abl = {(l, h): A.copy_fidelity(model, tokens, periods, np.random.default_rng(seed + 1), A.zero_ablate(l, h))
                for l in range(model.layers) for h in range(model.H)}
    profile = [A.period_offset_profile(cache, l, periods) for l in range(model.layers)]
    uniform = A.uniform_attention_baseline(periods, tokens.shape[1])
    return dict(nll=nll, acc=acc, prev_token=prev, induction=ind, ablate_zero=zero, ablate_mean=mean,
                dla=dla, dla_total=total, patch=patch, k_comp=k, k_comp_random=k_ref,
                copy=copy, copy_ablated=copy_abl, offset_profile=profile, uniform_baseline=uniform)


def format_report(r, layers, heads):
    L = []
    L.append("model: nll %.3f, accuracy %.3f on predictable positions" % (r["nll"], r["acc"]))
    L.append("")
    L.append("copy test: model follows an edit to the true source token in %.0f%% of cases (n=%d); "
             "edit to a different position: %.0f%%" % (100 * r["copy"]["follow"], r["copy"]["n"], 100 * r["copy"]["control"]))
    L.append("uniform-attention baseline for the induction score: %.3f" % r["uniform_baseline"])
    L.append("")
    L.append("head   prev-token  induction  d_nll(zero)  acc(zero)  d_nll(mean)  patch-recovery  copy-follow(zero)")
    zero = {(x["layer"], x["head"]): x for x in r["ablate_zero"]["rows"]}
    mean = {(x["layer"], x["head"]): x for x in r["ablate_mean"]["rows"]}
    pat = {(x["layer"], x["head"]): x for x in r["patch"]["rows"]}
    for l in range(layers):
        for h in range(heads):
            L.append("L%dH%d   %9.3f  %9.3f  %11.3f  %9.3f  %11.3f  %14.2f  %17.2f" % (
                l, h, r["prev_token"][l, h], r["induction"][l, h], zero[l, h]["d_nll"], zero[l, h]["acc"],
                mean[l, h]["d_nll"], pat[l, h]["recovery"], r["copy_ablated"][(l, h)]["follow"]))
    L.append("")
    L.append("direct logit attribution (correct-token logit, centred; total %.2f):" % r["dla_total"])
    for k, v in sorted(r["dla"].items(), key=lambda kv: -abs(kv[1])):
        L.append("  %-28s %7.2f" % (k, v))
    L.append("")
    L.append("attention by offset from the previous occurrence of the current token (key = t-p+o), o = -3..+3:")
    for l in range(layers):
        for h in range(heads):
            L.append("  L%dH%d  " % (l, h) + " ".join("%.2f" % x for x in r["offset_profile"][l][h]))
    L.append("")
    L.append("K-composition L0 head (row) -> L1 head (col), random-matrix reference %.3f:" % r["k_comp_random"])
    for i in range(heads):
        L.append("  L0H%d  " % i + "  ".join("%.3f" % x for x in r["k_comp"][i]))
    return "\n".join(L)
