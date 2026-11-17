# ntk

The neural tangent kernel in NumPy: a ReLU MLP with a hand-written per-sample backward pass, its empirical NTK, the closed-form infinite-width NTK and NNGP, and the exact kernel-regression solution that gradient descent follows in the lazy limit.

The question I wanted to poke at is how true "a wide network is just kernel regression" actually is, and where it stops being true.

```bash
cd ntk
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                    # ~30-70 s
PYTHONPATH=src python -m tangent_kernel convergence   # also: lazy, ensemble, spectral, alpha, depth
python make_results.py                                # rewrites results/, ~8 min
```

```python
import numpy as np
from tangent_kernel import MLP, ntk, gd_predict, stable_lr, train_gd

phi = np.linspace(0, 2 * np.pi, 16, endpoint=False)
X = np.stack([np.cos(phi), np.sin(phi)], 1)
y = np.sin(phi) + 0.5 * np.sin(3 * phi)

net = MLP(in_dim=2, width=1024, depth=2, seed=0)
K = net.ntk(X)
lr = stable_lr(K, 0.5)
trained = train_gd(net, X, y, lr, steps=400)
_, lin = gd_predict(K, K, net(X), net(X), y, lr, steps=400)
print(np.abs(trained.predict(X) - lin).max())   # tiny
```

## Results

All synthetic: points on a circle, 1-D target, depth-2 nets.

- The empirical kernel converges to the analytic one with slope -0.43 in width (theory says -0.5): 46% error at width 16, 5% at 4096.
- Wide nets are lazy, narrow ones aren't. At width 8 the kernel moves 83% during training, at width 2048 about 1%. The gap between the trained net and its linearisation goes from 0.31 to 0.018.
- An ensemble of 24 width-512 nets lands within 2.5% of the analytic NTK regression mean. A single net is 5.5% off.
- Spectral bias is very clean. Low frequencies get fit orders of magnitude faster, in exactly the order of the kernel eigenvalues. The interesting part is late in training on high frequencies: the real net is noticeably *slower* than its linearisation (0.30 vs 0.07 of the target left unfit at k=8). The kernel only moves about 1%, but it moves in exactly the directions that matter for the hard modes.
- Scaling the output by alpha pushes the net into the lazy regime as expected. In this setup the feature-learning end had slightly better test loss (0.0020 vs 0.0031).
- Deeper ReLU NTKs correlate inputs more and flatten the spectrum (condition number 475 at 1 hidden layer, 109 at 8).
- Kernel ridge with closed-form leave-one-out picks a regulariser within ~20% of the oracle. Plain interpolation of noisy data costs 2-3x the test error.

Numbers are in `results/*.json`. Small problems and 6-24 seeds, so these are trends, not tight intervals.
