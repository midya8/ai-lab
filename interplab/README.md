# interplab

Mechanistic interpretability from scratch on NumPy: train a tiny transformer on an in-context copying task, then take it apart with the standard circuit-analysis toolkit — attention-pattern scores, head ablation, direct logit attribution, activation patching, weight-based composition scores and a causal copy test — and find out what it actually learned.

It builds on [`microgrid`](../microgrid) (autodiff + `MiniGPT`) for training and re-implements the forward pass in plain NumPy with hook points (`InterpModel`). A test checks that its logits match the microgrid model to 1e-9, so every analysis is of the trained network itself.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                       # 20 tests, ~25 s
PYTHONPATH=src python -m interplab --seed 0 --steps 4000 # train (~100 s) and print the circuit report
PYTHONPATH=src python -m interplab --load model.npz      # re-analyse saved weights (--save model.npz when training)
```

## The task

Each sequence repeats a block of `p` distinct tokens, `p` drawn per sequence from 4..8 (vocab 24, length 17). The first period is unpredictable; afterwards the next token is determined by the context. Because `p` varies, "copy the token k steps back" does not work — the model has to find where the current token occurred before and copy what followed it. That is the induction-head setting from the interpretability literature, which makes the textbook circuit the natural hypothesis to test.

## The model: 2 layers, 2 heads each, d=32 (MLPs included), 4000 steps of AdamW

All three seeds reach 98–99% accuracy on predictable positions (NLL 0.13–0.16; chance is 3.18). Full output: [`experiments/results_seeds_0_1_2.txt`](experiments/results_seeds_0_1_2.txt).

| seed | acc | copy test: follows edit to true source | edit elsewhere (control) |
|---|---|---|---|
| 0 | 0.992 | 99% | 17% |
| 1 | 0.986 | 99% | 20% |
| 2 | 0.981 | 96% | 17% |

The **copy test** (`copy_fidelity`) is the cleanest causal evidence: pick a query position with exactly one earlier copy of the answer, overwrite that copy with a token that appears nowhere else, and see whether the model's top prediction becomes the new token. It does, 96–99% of the time. The model really is copying from the right place in context. (The 17–20% control rate is not zero: editing another position in the first period sometimes also moves the prediction, so the model's lookup is not perfectly sharp.)

## What I expected vs. what I found

Expected (textbook): a layer-0 previous-token head feeding a layer-1 induction head (attends to the token *after* the earlier occurrence of the current token), visible as high K-composition between them.

Found, consistently across all three seeds:

* **No textbook induction head.** Layer-1 heads score 0.01–0.07 on the induction score, *below* the 0.096 a uniform-attention head would get. Layer-1 heads instead put most weight on the previous token (0.2–0.37).
* **K-composition from layer 0 to layer 1 is at the random-matrix level** (0.17–0.21 vs 0.177 for random matrices). The weight-based signature of the classic circuit is absent. (Caveat: this ignores LayerNorm and the MLP in between, so it is evidence of absence only of the *direct* path.)
* **The lookup happens in layer 0.** Zero-ablating either layer-0 head drops accuracy from ~0.99 to 0.22–0.44 and the copy test from ~98% to 18–40%; ablating a layer-1 head costs much less (accuracy 0.80–0.95, copy 70–94%). Layer-0 heads' attention is elevated around the right key (offset +1 from the previous occurrence: 0.22–0.27 against the 0.096 uniform level) but smeared over offsets −2..+2, i.e. soft, not a sharp pointer.
* **MLPs do most of the direct writing to the correct logit** (layer-1 MLP: 6.0–6.9 of a total 11.5–13.7 logit units; the two layer-1 heads together 1.2–5.6). On seed 2 the layer-0 MLP contributes 3.8 and the layer-1 heads almost nothing, so the division of labour differs by seed.
* Mean-ablation hurts less than zero-ablation for layer-0 heads (e.g. seed 0: ΔNLL 2.0 vs 3.0), meaning part of what zero-ablation breaks is "output has the wrong scale", not only "information is missing".
* Activation patching (clean head output into a corrupted run) recovers 0.5–0.8 of the clean log-prob for layer-0 heads on seeds 0–1; on seed 2 the layer-1 heads recover only 0.12–0.14. Recovery numbers are not additive across heads.

So: the network solves the task, provably by copying from the right source, but with a distributed mechanism in which layer-0 attention and the MLPs carry the lookup. I did **not** reverse-engineer it fully; I can say where the load is carried and where it isn't, not the exact algorithm. A natural next step would be to patch/ablate MLP neurons and test whether a position-plus-content "soft match" in layer 0 explains the smeared attention.

## What is in the box

| module | what |
|---|---|
| `task.py` | periodic-sequence generator, predictable-position mask, corrupted batches for patching |
| `model.py` | `InterpModel`: NumPy forward with hook points and activation cache, `from_minigpt`, save/load |
| `train.py` | `train_induction` using microgrid's autodiff |
| `analysis.py` | `previous_token_scores`, `induction_scores`, `period_offset_profile`, `head_ablation_table` (zero/mean), `logit_attribution` (exactly additive; tested), `patch_head_effects`, `composition_scores` (Q/K/V), `copy_fidelity`, `uniform_attention_baseline` |
| `report.py` / `__main__.py` | one-call report and CLI |

## Honest limits

* Three seeds, 2×2 heads, one synthetic task. Per-head numbers differ across seeds; only the pattern (layer 0 carries the lookup, no sharp induction head) is consistent.
* Direct logit attribution freezes the final LayerNorm scale and ignores indirect effects (a layer-0 head helping a layer-1 head shows up in the later component).
* Ablations use batches of 256 sequences from a fixed seed; no confidence intervals.
* The task, model size and training length were fixed before looking at any analysis; I did add the offset profile, the uniform baseline and the copy test after the first report showed no textbook induction head.
