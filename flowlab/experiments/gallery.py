"""ASCII gallery: data vs samples from an 8-coupling flow, on the ring and the checkerboard."""
import sys
sys.path.insert(0, "../src")
from flowlab import DATASETS, Flow, ascii_density, fit

for name, lim in (("ring8", 3.2), ("checker4", 2.4)):
    ds = DATASETS[name]()
    f = Flow(2, 8, 64, seed=0)
    fit(f, ds.sample(20000, 100), steps=3000, seed=0)
    a = ascii_density(ds.sample(20000, 5), lim, 40, 18).split("\n")
    b = ascii_density(f.sample(20000, seed=5), lim, 40, 18).split("\n")
    print(f"== {name}: data (left) vs flow samples, 8 affine couplings (right)")
    for x, y in zip(a, b):
        print(f"|{x}|  |{y}|")
    print()
