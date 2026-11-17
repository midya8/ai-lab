"""moe: sparse Mixture-of-Experts routing from scratch in NumPy."""

from .data import make_task, sample_task
from .layers import DenseFFN, MoELayer
from .metrics import load_stats, routing_nmi, routing_purity
from .model import Net
from .train import fit

__all__ = [
    "make_task", "sample_task", "DenseFFN", "MoELayer", "load_stats",
    "routing_nmi", "routing_purity", "Net", "fit",
]
