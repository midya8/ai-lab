# hopfield

Classical Hopfield networks (Hebbian, pseudoinverse and Storkey rules) next to modern ones: dense associative memories with polynomial or exponential energies, and the continuous version whose update is literally softmax attention.

I built it to see the capacity difference for myself, and to understand what the attention temperature is actually doing.

```bash
cd hopfield
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q          # 34 tests, ~1 s
PYTHONPATH=src python -m hopfield glyphs    # also: capacity, stress, basin, correlated, beta
python make_results.py                      # rewrites results/, ~45 s
```

```python
import numpy as np
from hopfield import random_patterns, flip_noise, dam_retrieve

rng = np.random.default_rng(0)
X = random_patterns(400, 100, rng)          # 400 memories in 100 neurons
q = flip_noise(X[:5], 0.15, rng)
out = dam_retrieve(X, q, kind="exp", beta=12.0, rng=rng).states
print((out == X[:5]).all(axis=1))           # all True
```

## Results

Exact recovery rate with 10% of bits flipped, N=100 neurons, 3 seeds:

| patterns | Hebbian | pseudoinverse | Storkey | dense, cubic | dense, exp |
|--:|--:|--:|--:|--:|--:|
| 14 | 0.69 | 1.00 | 1.00 | 1.00 | 1.00 |
| 30 | 0.00 | 1.00 | 0.94 | 1.00 | 1.00 |
| 50 | 0.00 | 0.47 | 0.17 | 1.00 | 1.00 |
| 400 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 |

Hebbian falls off around 0.14N, as the textbooks say. With 25% noise, the quintic and exponential memories still recover 88-93% of patterns at 4000 patterns (40N); cubic gives up around 4N.

Correlated patterns are what really hurts. With a shared template, Hebbian is dead by moderate overlap and the cubic memory follows, while the exponential memory stays perfect.

The attention part: with 32 continuous patterns in 64 dimensions there's a sharp switch between beta 6 and 9. Below it the only stable point is the average of all 32 memories. One step even looks like it's converging, then iterating drifts back to the mean. Above it every memory is its own fixed point and one step is enough. So a soft attention head averages values and a sharp one retrieves.

Tables are in `results/`. Synthetic patterns, small N and 3 seeds, so treat differences of 0.05 as noise.
