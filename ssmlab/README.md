# ssmlab

**A Mamba-style selective state-space model from scratch in NumPy, with hand-written backprop through the scan, and an experiment that isolates what "selective" buys.** No autodiff, no deep-learning framework: the forward pass, the backward pass (itself a reverse-time scan), the optimiser and the synthetic tasks are all here, and every gradient is checked against finite differences.

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                                   # 31 tests, ~8 s
PYTHONPATH=src python -m ssmlab train --steps 1500 --seq-len 16 --save /tmp/sel.npz   # ~35 s
PYTHONPATH=src python -m ssmlab train --lti --steps 1500 --seq-len 16                 # the fixed-dynamics baseline
PYTHONPATH=src python -m ssmlab eval /tmp/sel.npz --seq-len 32                        # length generalisation
```

## What is in the box

| Piece | File | Notes |
|---|---|---|
| Linear-recurrence scan | `scan.py` | `h_t = a_t h_{t-1} + b_t` sequentially and as a log-depth Hillis-Steele associative scan; `reverse_scan` is the exact adjoint (checked by a dot-product test), so backprop through time is one more scan |
| Diagonal SSM layer | `layers.py` | `A = -exp(A_log)`, zero-order-hold for `A`, Euler for `B` (as in the Mamba reference code). **Selective**: step size `delta`, `B` and `C` are functions of the input. **LTI** (S4D-style): they are fixed parameters, so the layer is exactly a convolution (`SSM.kernel`, tested against the closed form) |
| Block and model | `model.py` | RMSNorm, in-projection, causal depthwise conv + SiLU, SSM, SiLU gate, out-projection, residual; embedding and head. `step()` decodes one token with a fixed-size state and matches the full forward to 1e-10; `generate`, `save`/`load` |
| Training | `optim.py`, `train.py` | AdamW (decay on matrices only), global-norm clipping, warmup + cosine schedule, fresh samples every step, deterministic per seed |
| Tasks | `tasks.py` | selective copying (symbols scattered among noise, emitted in order at the end) and associative recall |

What the tests pin down: scan equivalence for 6 lengths with and without an initial state; the reverse scan as the exact adjoint; finite-difference gradients for every parameter of the SSM layer (both modes), RMSNorm, the conv, and the **whole model** (every one of the 20+ parameter tensors, both modes); a test that the harness **fails** when the SSM backward is deliberately corrupted; causality (changing future tokens never changes earlier logits); step-vs-full-forward parity; the LTI layer equals its closed-form kernel and is linear in its input while the selective one is not; recurrent state size is independent of length; masked positions receive zero gradient; checkpoint round trips with a shape check; training is deterministic.

## Experiment: does selectivity matter? (selective copying)

`experiments/lti_vs_selective.py`. Task: 4 symbols (from 4) are scattered at random positions among L noise tokens; after the noise, 4 marker tokens ask for them back in order. Chance is 0.25 per token. Two 2-layer, d_model-16 models that differ **only** in the SSM mixer (selective vs fixed `delta`, `B`, `C`), 2,500 steps of 32 fresh samples, 3 seeds each, 1,000 fresh test sequences per seed. "2L" means the same trained model scored on a noise length **twice as long as it ever saw**, no retraining. Cells are the mean over 3 seeds, with the range of the 3 in brackets.

| noise length L | model | params | token acc | exact (all 4 right) | token acc at 2L | exact at 2L |
|---|---|---|---|---|---|---|
| 16 | selective | 7,344 | **1.000** [1.000-1.000] | **1.000** [0.999-1.000] | 0.991 [0.990-0.992] | 0.962 [0.958-0.966] |
| 16 | LTI | 5,296 | 0.909 [0.888-0.942] | 0.680 [0.607-0.804] | 0.619 [0.594-0.644] | 0.110 [0.078-0.138] |
| 32 | selective | 7,344 | **1.000** [0.999-1.000] | **0.999** [0.997-1.000] | 0.993 [0.984-0.999] | 0.972 [0.934-0.997] |
| 32 | LTI | 5,296 | 0.867 [0.855-0.883] | 0.536 [0.482-0.596] | 0.549 [0.499-0.579] | 0.077 [0.046-0.099] |
| 64 | selective | 7,344 | **1.000** [1.000-1.000] | **0.999** [0.998-1.000] | 0.999 [0.998-1.000] | 0.996 [0.991-0.999] |
| 64 | LTI | 5,296 | 0.810 [0.763-0.849] | 0.389 [0.267-0.487] | 0.519 [0.488-0.537] | 0.056 [0.047-0.072] |

Reading it, with the caveats:

* The selective model solves the task at every length and **extrapolates**: trained at noise length 16 it is still 99% per token (96% of whole sequences) at 32; trained at 64 it is 99.6% exact at 128.
* The LTI model degrades as the noise gets longer (0.91 -> 0.87 -> 0.81 per token) and **collapses off-distribution** (0.52-0.62 per token at 2L, whole-sequence accuracy 5-11%). A fixed convolution cannot "skip the noise"; it learns a length-specific shortcut.
* **The LTI model above is smaller** (no `W_dt`/`W_B`/`W_C`). `experiments/param_matched.py` widens it to d_model 20 (7,580 params, slightly *more* than the selective model) at L=32: token accuracy 0.904 [0.882-0.942], exact 0.654 [0.584-0.788], 0.604 at 2L (exact 0.120). Extra width helps a little and does not close the gap, so size is not what explains the difference.
* The LTI block here still has a gate and a conv, so it is not a purely linear model; that is why it scores 0.8-0.9 rather than chance. The claim is only that input-dependent `delta`/`B`/`C` is what makes the difference *within this block*.
* Same 2,500-step budget for everyone and no tuning of either model; 3 seeds is enough to see a gap this large, not to rank small differences. Synthetic task, tiny models: this says nothing about language-model scale.

## Experiment: scan and state size (`experiments/scan_benchmark.py`)

The associative scan is exact (max difference 4e-15) but **slower than the plain loop in NumPy** at every length tried (e.g. 88 ms vs 10 ms at T=2048): it does O(T log T) work in a single thread of vectorised ops. Its value is depth, which only pays off on parallel hardware, so the model defaults to the sequential scan and offers `scan_method="parallel"` (tested to give identical outputs and gradients).

The recurrent state of the 4-layer, d_model-64 model is 77,824 bytes per sequence at any length. A transformer KV cache of the same width and depth would hold 524 KB at T=128 and 33.5 MB at T=8192 (6.7x and 431x larger); that comparison is arithmetic, not a measured transformer.

## Layout

```
src/ssmlab/   scan.py layers.py model.py optim.py train.py tasks.py __main__.py
tests/        scan, gradient checks, parity/causality/linearity, training, CLI
experiments/  lti_vs_selective.py (+ json/logs), param_matched.py (+ jsonl), scan_benchmark.py (+ json)
```

## Limitations

Float64 NumPy on CPU, so the models are tiny (about 7k parameters). The zero-order-hold/Euler discretisation is the Mamba shortcut, not an exact ZOH for `B`. There is no complex-valued or `A`-structured (HiPPO) initialisation, no chunked scan, and no language-model data. Associative recall (`tasks.associative_recall`, `--task recall`) is implemented and tested for shape/labels but I have not run a study on it.
