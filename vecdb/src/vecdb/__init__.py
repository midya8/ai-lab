"""vecdb: approximate nearest-neighbour search (flat, IVF-Flat/PQ, HNSW) and a small vector DB."""
from .collection import Collection, Hit
from .datasets import clustered, gaussian, queries_like, recall_at_k
from .filters import compile_filter
from .flat import FlatIndex
from .hnsw import HNSWIndex
from .io import load_index, save_index
from .ivf import IVFIndex
from .pq import ProductQuantizer

__all__ = ["Collection", "Hit", "FlatIndex", "IVFIndex", "HNSWIndex", "ProductQuantizer",
           "compile_filter", "save_index", "load_index", "clustered", "gaussian", "queries_like",
           "recall_at_k"]
__version__ = "0.1.0"
