"""boptlab: Bayesian optimisation from scratch in NumPy (GP, EI/PI/UCB/Thompson, batch, benchmarks)."""
from .gp import GP, Kernel, fit_gp, log_marginal_likelihood
from .acquisition import expected_improvement, probability_of_improvement, confidence_bound, mc_expected_improvement
from .optimizer import BayesOpt, random_search, halton, optimize_acquisition
from .benchmarks import BENCHMARKS, Benchmark, branin, hartmann3, hartmann6, ackley, levy, rosenbrock, noisy

__all__ = [n for n in dir() if not n.startswith("_")]
