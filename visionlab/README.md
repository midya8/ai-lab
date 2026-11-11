# visionlab

A convolutional neural network, its trainer, three attribution methods and two adversarial attacks, all in **pure NumPy** with hand-written backpropagation. No PyTorch, no autograd: every layer's backward pass is checked against finite differences.

![saliency / smoothgrad / Grad-CAM / occlusion on six shapes](docs/explain.png)

*Columns: input · saliency · SmoothGrad · Grad-CAM · occlusion. Rows: one image per class (ring, square, disk, cross, triangle, stripes).*

## What's in it

| Module | What it does |
|---|---|
| `layers.py` | `Conv2D` (im2col via strided window view + one matmul), `MaxPool2D` (one-argmax gradient routing), `Dense`, `BatchNorm2D` (batch stats in training, running stats in eval, saved with the model), `GlobalAvgPool`, `ReLU`, inverted `Dropout`, `Flatten`, fixed `Normalize` |
| `model.py` | `Sequential` that records every layer's activation and output-gradient (used by Grad-CAM), softmax CE with label smoothing, `.npz` save/load with the architecture embedded |
| `optim.py` | SGD + momentum, AdamW-style decoupled weight decay |
| `data.py` | Procedural, anti-aliased, seeded dataset: disk, ring, square, triangle, cross, stripes at random size/position/rotation/brightness/noise |
| `train.py` | Mini-batch trainer: cosine LR, flip/rot90 augmentation, optional FGSM adversarial training |
| `explain.py` | Saliency, SmoothGrad, Grad-CAM, occlusion, plus a **deletion-curve faithfulness metric** to compare them |
| `attacks.py` | FGSM and PGD (random start, L-inf ball) in raw pixel space |
| `certify.py` | **Interval bound propagation**: sound logit bounds for every L-inf perturbation within eps, margin-based certification, certified radius by bisection, and the backward pass that lets the bounds be trained through (IBP training) |
| `robust.py` | Accuracy under noise / translation / contrast loss and under attack |
| `png.py` | Dependency-free PNG writer and heat-map montage |

## Quick start

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python -m visionlab train --epochs 12 --out model.npz      # ~10 s on a laptop CPU, 38,566 parameters
python -m visionlab eval --model model.npz                  # accuracy + Wilson CI + confusion matrix
python -m visionlab explain --model model.npz --faithfulness 150
python -m visionlab robust --model model.npz
python -m visionlab show --model model.npz -n 2             # ASCII-art predictions
python -m visionlab train --head gap --batchnorm --width 16   # translation-robust head (see below)
python -m visionlab train --ibp-eps 0.03 --epochs 16 --out cert.npz   # certified-robustness training (~1 min)
python -m visionlab certify --model cert.npz --eps 0.01 0.02 0.03 0.05 --radius 100
python -m pytest -q                                         # 64 tests, ~35 s
```

## Results (all on synthetic shapes, test set disjoint from training)

**Accuracy.** 12 epochs, 4,000 training images: **97.3 %** on 1,000 test images (95 % Wilson CI 96.1-98.1 %). The errors are mostly disk <-> square (15 of 27) and square <-> triangle (9).

**Which explanation can you trust?** Deletion AUC = mean class probability while the top 0-40 % of pixels ranked by each map are blanked (lower means the map found what the model actually uses). 147 correctly classified images:

| Method | Deletion AUC |
|---|---|
| occlusion | **0.280** |
| SmoothGrad | 0.345 |
| plain saliency | 0.373 |
| Grad-CAM | 0.588 |
| random ranking | 0.691 |

Occlusion wins because it measures the effect of removing pixels directly, which is also what the metric does, so it is favoured by construction. Grad-CAM's maps are visually the most plausible (see image) but only 12x12 resolution, so they rank individual pixels poorly. All four beat random.

**Brittleness.** On the same model: 90 % accuracy at pixel noise sigma 0.2, but only 79 % when the image is translated by 2 px diagonally and 37 % at 4 px. The dense head is not translation invariant and training never shows shifted images beyond the dataset's own jitter. Gradient attacks are far more damaging than noise: FGSM at eps = 0.05 (about 13/255) cuts accuracy to 36 %, 10-step PGD to 26 %.

**Adversarial training** (`experiments/adv_training.py`, mean of 3 seeds, FGSM eps = 0.05 mixed 1:1 into every batch; attacks on 400 test images):

| | clean | FGSM 0.02 | FGSM 0.05 | FGSM 0.1 | PGD 0.05 | PGD 0.1 |
|---|---|---|---|---|---|---|
| standard | 0.986 | 0.689 | 0.327 | 0.128 | 0.186 | 0.031 |
| adversarially trained | 0.982 | 0.892 | 0.723 | 0.483 | 0.652 | 0.265 |

Clean accuracy is unchanged within seed noise (+-0.01); robustness at and below the training budget roughly doubles. Noise/shift robustness did not improve (noise 0.3: 0.757 vs 0.750; shift 2 px: 0.795 vs 0.805), and contrast-0.5 accuracy dropped (0.910 -> 0.865). Robustness does not transfer across perturbation types, and eps = 0.1 stays largely broken. Only 3 seeds, so differences of ~0.01-0.02 are noise.

**Where the dense head breaks, and a global-average-pool head** (`experiments/heads_shift.py`, mean of 3 seeds, 1,000 test images; shifts are diagonal with zero fill). `--head gap` replaces flatten + MLP with a third conv and a global average pool, so evidence is summed over positions instead of being tied to them; `--batchnorm` adds `BatchNorm2D` after every conv.

| head | params | clean | shift 1px | 2px | 3px | 4px | noise 0.3 | contrast 0.5 |
|---|---|---|---|---|---|---|---|---|
| dense | 38,566 | 0.984 (±0.008) | 0.944 | 0.792 | 0.557 | 0.366 | 0.745 | 0.902 |
| dense + BN | 38,614 | 0.966 (±0.003) | 0.929 | 0.746 | 0.498 | 0.310 | 0.730 | 0.880 |
| GAP | 6,086 | 0.894 (±0.013) | 0.884 | 0.858 | 0.833 | 0.787 | 0.447 | 0.369 |
| GAP + BN | 6,198 | 0.937 (±0.009) | 0.937 | 0.918 | 0.901 | 0.873 | 0.597 | 0.523 |
| GAP + BN, width 16 | 23,910 | 0.978 (±0.003) | 0.978 | 0.970 | 0.961 | 0.928 | 0.716 | 0.656 |

The dense head loses 62 points between 0 and 4 px of shift; the width-16 GAP + BN model loses 5 and, with 38 % fewer parameters, is within noise of the dense model on clean data. BatchNorm is what makes the small GAP net trainable in 12 epochs (+4 points clean) but it does *not* help the dense head (-2 points, one fewer point of noise robustness). The price of the GAP head is photometric brittleness: at contrast 0.5 it keeps 0.656 against the dense model's 0.902, and noise 0.3 is about the same. A plausible cause is that averaging over the whole image makes the decision depend on global activation level, which dimming changes; I did not test this explanation. The comparison is not parameter-matched except for the last row, only 12 epochs and 3 seeds were run, and "clean" is equal for dense and the last GAP row only within seed noise.

**Contrast augmentation: fixes the family it trains on, not contrast in general** (`train(..., photometric=p)` multiplies each training image by c ~ U(1-p, 1); `experiments/photometric.py`, GAP + BN width 16, mean of 3 seeds, 1,000 test images).

| photometric | clean | x0.5 | x0.3 | about-mean 0.5 | about-mean 0.3 |
|---|---|---|---|---|---|
| 0 | 0.978 | 0.656 | 0.171 | 0.403 | 0.179 |
| 0.5 | 0.967 | 0.958 | 0.583 | 0.170 | 0.167 |
| 0.7 | 0.960 | 0.962 | 0.917 | 0.168 | 0.167 |

Scaling by c (the `contrast` column of the robustness report) goes from 0.656 to 0.96 at x0.5 for about 1-2 points of clean accuracy. But that is the very perturbation trained on. On an unseen one, contrast about the mean ((x-0.5)*c+0.5, which also lifts the dark background), accuracy is *worse* than the baseline (0.403 -> 0.170, chance is 0.167): the network learned "dim means shrink", not contrast invariance. Treat the x-c numbers as in-distribution; real photometric robustness needs augmentation that covers brightness and offset too (not done here).

**Certified robustness by interval bound propagation** (`certify.py`, `experiments/certified_robustness.py`, mean of 3 seeds, 800 test images, raw output in `experiments/certified_robustness.out.txt`). An attack can only show a model is *broken*; a bound can show it is *safe*. The input box [x-eps, x+eps] (clipped to [0,1]) is pushed through the network as a centre and radius per activation, giving sound bounds on every logit; an image is **certified** when the lower bound of `logit_y - logit_j` is positive for all `j`. Certified accuracy is a proven lower bound on accuracy under *any* L-inf attack of that size; PGD accuracy is an upper bound, so the true robust accuracy sits between them.

| model | clean | certified @0.01 | @0.02 | @0.03 | @0.05 | PGD-20 @0.03 | median certified radius |
|---|---|---|---|---|---|---|---|
| standard | 0.988 | 0.000 | 0.000 | 0.000 | 0.000 | 0.479 | 0.0011 |
| FGSM adv-trained (0.05) | 0.982 | 0.037 | 0.000 | 0.000 | 0.000 | 0.859 | 0.0037 |
| IBP-trained (eps 0.03) | 0.839 | 0.766 | 0.667 | 0.561 | 0.305 | 0.718 | 0.0363 |

Standard and FGSM-trained networks are **not certifiable at all** with interval bounds, even when PGD says the FGSM model keeps 86 % at eps = 0.03: the bound is loose for networks that were never trained to be tight, so "no certificate" does not mean "attackable" (the gap is the point of the table). Training through the bounds (loss = 0.5 clean CE + 0.5 worst-case CE, eps and the mix ramped over 40 % of the steps) proves 56 % of test images safe at eps = 0.03 and 31 % at 0.05, at a price of about 15 points of clean accuracy, and the seed-to-seed spread of clean accuracy is large (std 0.045). Its PGD accuracy at 0.03 (0.72) is below the adversarially trained model's (0.86): certifiability was bought with raw accuracy, not for free. The margin-based certificate is only slightly tighter than comparing logit intervals (0.561 vs 0.555 at 0.03 for the IBP model) because the last layer is a single 64-wide dense map.

Honest caveats: the training recipe (16 epochs, eps 0.03, kappa 0.5) was picked from a short sweep of five settings scored on a different test seed (999; the table uses 998), so the IBP row had a little tuning that the others did not; widening the network (16 channels) did not help in that sweep. Certificates are exact in real arithmetic; the float32 model rounds, so a result within ~1e-6 of the boundary is not guaranteed. Three seeds only.

## How it is tested

- Gradient checks (float64 finite differences) for `Conv2D` at several paddings/kernel sizes, `Dense`, `ReLU`, `MaxPool2D` (even and odd sizes), `Normalize`, the cross-entropy + label-smoothing gradient, the whole network w.r.t. every parameter *and* its input, and the loss gradient used by the attacks.
- `Conv2D` equals a naive 4-loop convolution; max-pool ties send the gradient to exactly one pixel.
- Saliency equals the finite-difference logit gradient; Grad-CAM uses the right layer and is class-specific; occlusion of blank background is zero; the deletion curve starts at the clean probability and an informative map beats a random one.
- FGSM/PGD stay inside the eps ball and the valid pixel range and raise the loss; adversarial training improves FGSM accuracy; optimisers converge on a quadratic.
- `BatchNorm2D` gradient-checked in training and eval mode, running statistics converge to the data's and do not move in eval mode; `GlobalAvgPool` gradient-checked and invariant to any spatial permutation; whole-network gradient check for the GAP + BN architecture in both modes; BN running statistics survive save/load; the default architecture's parameter count is pinned (38,566) so the new options cannot silently change it.
- Interval bounds: equal the logits at eps = 0; enclose 80+ random points, box corners and a PGD attack for four architectures (dense, GAP, GAP + BN, dense + BN, with negative BatchNorm slopes); widen monotonically with eps; are *exact* for a single linear layer; the margin bound is sound and never looser than the naive interval comparison; certified radii bracket the certification boundary; no certified image is flipped by a 20-step PGD attack.
- IBP training: the worst-case loss gradient matches float64 finite differences for every parameter (dense and GAP heads); at eps = 0 it equals plain cross-entropy and its gradient; IBP training certifies >10 points more than standard training; unsupported combinations (BatchNorm, FGSM + IBP) raise.
- Dataset determinism, class balance, label-preserving augmentation, PNG round trip, save/load round trip.

## Limits

- The data is procedural and the classes are easy for humans; numbers do not transfer to natural images.
- The deletion metric replaces pixels with 0 (background), which is itself off-distribution for the network; treat the ranking as indicative, not definitive.
- Single train/test split per CLI run; the experiment uses 3 seeds.
- FGSM-trained models are known to be partly "gradient masked"; PGD is a stronger check but still an attack. Only the `certify` numbers are proofs, and only for L-inf perturbations (not shifts or contrast), for the dense / GAP architectures here, on synthetic images.
- IBP bounds are loose on deeper or wider networks and IBP training is not available with BatchNorm; tighter relaxations (CROWN, zonotopes) are not implemented.
