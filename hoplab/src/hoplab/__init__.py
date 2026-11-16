"""hoplab: classical and modern Hopfield networks (associative memories) from scratch in NumPy."""

from .classical import (
    hebbian,
    pseudoinverse,
    storkey,
    energy,
    retrieve,
)
from .modern import (
    dam_energy,
    dam_retrieve,
    softmax,
    cont_update,
    cont_retrieve,
    cont_energy,
    attention,
    effective_number,
    separation,
)
from .patterns import random_patterns, correlated_patterns, flip_noise, mask_noise, overlap
from .glyphs import DIGITS, digit_patterns, render

__all__ = [
    "hebbian", "pseudoinverse", "storkey", "energy", "retrieve",
    "dam_energy", "dam_retrieve", "softmax", "cont_update", "cont_retrieve",
    "cont_energy", "attention", "effective_number", "separation",
    "random_patterns", "correlated_patterns", "flip_noise", "mask_noise", "overlap",
    "DIGITS", "digit_patterns", "render",
]
__version__ = "0.1.0"
