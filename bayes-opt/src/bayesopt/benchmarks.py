"""Standard global-optimisation test functions (minimisation) with known optima."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Benchmark:
    name: str
    bounds: np.ndarray          # (d, 2)
    f: callable
    optimum: float
    argmins: np.ndarray         # (k, d) known global minimisers

    @property
    def dim(self):
        return len(self.bounds)

    def __call__(self, x):
        return float(self.f(np.asarray(x, float)))


def _branin(x):
    a, b, c, r, s, t = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi, 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x[1] - b * x[0] ** 2 + c * x[0] - r) ** 2 + s * (1 - t) * np.cos(x[0]) + s


_A3 = np.array([[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]], float)
_P3 = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470], [1091, 8732, 5547], [381, 5743, 8828]], float)
_A6 = np.array([[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]], float)
_P6 = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886], [2329, 4135, 8307, 3736, 1004, 9991],
                       [2348, 1451, 3522, 2883, 3047, 6650], [4047, 8828, 8732, 5743, 1091, 381]], float)
_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])


def _hartmann(A, P):
    return lambda x: -float(_ALPHA @ np.exp(-(A * (x[None, :] - P) ** 2).sum(1)))


def _ackley(x):
    d = len(x)
    return -20 * np.exp(-0.2 * np.sqrt((x**2).mean())) - np.exp(np.cos(2 * np.pi * x).mean()) + 20 + np.e


def _levy(x):
    w = 1 + (x - 1) / 4
    return (np.sin(np.pi * w[0]) ** 2 + ((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2)).sum()
            + (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2))


def _rosenbrock(x):
    return float((100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2).sum())


branin = Benchmark("branin", np.array([[-5, 10], [0, 15]], float), _branin, 0.397887,
                   np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]]))
hartmann3 = Benchmark("hartmann3", np.array([[0, 1]] * 3, float), _hartmann(_A3, _P3), -3.86278,
                      np.array([[0.114614, 0.555649, 0.852547]]))
hartmann6 = Benchmark("hartmann6", np.array([[0, 1]] * 6, float), _hartmann(_A6, _P6), -3.32237,
                      np.array([[0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573]]))


def ackley(dim=5):
    return Benchmark(f"ackley{dim}", np.array([[-32.768, 32.768]] * dim), _ackley, 0.0, np.zeros((1, dim)))


def levy(dim=4):
    return Benchmark(f"levy{dim}", np.array([[-10, 10]] * dim, float), _levy, 0.0, np.ones((1, dim)))


def rosenbrock(dim=4):
    return Benchmark(f"rosenbrock{dim}", np.array([[-2, 2]] * dim, float), _rosenbrock, 0.0, np.ones((1, dim)))


BENCHMARKS = {"branin": branin, "hartmann3": hartmann3, "hartmann6": hartmann6,
              "ackley5": ackley(5), "levy4": levy(4), "rosenbrock4": rosenbrock(4)}


def noisy(bench: Benchmark, sd: float, rng) -> Benchmark:
    """Same landscape, observations corrupted by N(0, sd^2) noise (regret should be judged on the clean f)."""
    return Benchmark(bench.name + f"+noise{sd}", bench.bounds, lambda x: bench.f(x) + sd * rng.standard_normal(),
                     bench.optimum, bench.argmins)
