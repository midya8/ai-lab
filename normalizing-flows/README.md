# normalizing-flows

RealNVP and NICE style normalizing flows in NumPy, with ActNorm, permutations or a learned Glow-style linear mixing layer. Every layer has a hand-written backward pass that's checked against finite differences, including a test that deliberately breaks the logdet and makes sure the check notices.

I trained them on 2-D densities where the exact log-density is known (an 8-Gaussian ring and a checkerboard), so instead of a bare NLL I can report how many nats the flow is away from the truth.

```bash
cd normalizing-flows
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
cd experiments && OMP_NUM_THREADS=1 python depth_study.py ring8    # ~6 min
```

```python
from normflow import Flow, fit, gaussian_ring, nll_gap, ascii_density

ds = gaussian_ring()
f = Flow(d=2, n_layers=8, hidden=64, seed=0)
fit(f, ds.sample(20000, 0), steps=3000)
print(nll_gap(f, ds))                     # gap = KL(data || flow) in nats
print(ascii_density(f.sample(20000), lim=3.2))
```

## Results (3 seeds, 3000 steps)

Gap to the true density in nats, lower is better:

| couplings | ring, affine | ring, additive | checkerboard, affine | checkerboard, additive |
|--:|--:|--:|--:|--:|
| 2 | 1.20 | 1.54 | 0.59 | 0.70 |
| 4 | 0.60 | 1.12 | 0.48 | 0.61 |
| 8 | 0.50 | 0.64 | 0.38 | 0.52 |
| 16 | 0.40 | 0.43 | 0.32 | 0.47 |

On the ring, depth helps steadily but even 16 affine couplings leave 8% of samples stranded between the modes (an exact sampler would put 0.2% there). That's the cost of bending one connected Gaussian into eight blobs. Volume-preserving (additive) layers catch up with depth on the ring but not on the checkerboard, where the hard edges need the scale term.

The Glow-style learned mixing didn't help in 2-D. Which makes sense, since swapping two coordinates is already all the mixing you can do.

Results are in `experiments/*_results.txt` and `results/mixing.txt`.

Toy 2-D data, small MLPs, short training, so every gap is an upper bound. Next I'd try spline couplings and a mixture base distribution to get around the topology problem.
