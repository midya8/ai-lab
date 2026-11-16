"""gnnlab: graph neural networks from scratch in NumPy, with hand-written backprop."""

from .decoupled import APPNP, JKNet, SGC, label_propagation, propagate_ppr
from .graph import (
    Graph,
    GraphOps,
    build_ops,
    edge_homophily,
    make_split,
    sbm_graph,
)
from .layers import DenseLayer, GATLayer, GCNLayer, SAGELayer
from .metrics import accuracy, dirichlet_energy, mean_pairwise_cosine
from .model import GNN, softmax_xent
from .train import TrainResult, fit

__all__ = [
    "Graph", "GraphOps", "build_ops", "edge_homophily", "make_split", "sbm_graph",
    "DenseLayer", "GATLayer", "GCNLayer", "SAGELayer",
    "accuracy", "dirichlet_energy", "mean_pairwise_cosine",
    "APPNP", "JKNet", "SGC", "label_propagation", "propagate_ppr",
    "GNN", "softmax_xent", "TrainResult", "fit",
]
__version__ = "0.2.0"
