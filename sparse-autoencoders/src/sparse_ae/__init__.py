"""sparse_ae: sparse autoencoders from scratch in NumPy, tested on data whose true features are known."""

from .data import FeatureWorld, make_hierarchical_world, make_world
from .metrics import (
    absorption, coefficient_ratio, feature_recall, feature_recovery, fvu, l0, latent_purity, dead_latents,
)
from .sae import SAE
from .train import fit
from .variants import GatedSAE, JumpReLUSAE

__all__ = [
    "FeatureWorld", "make_world", "make_hierarchical_world", "JumpReLUSAE", "GatedSAE", "absorption", "SAE", "fit", "feature_recovery", "latent_purity",
    "fvu", "l0", "dead_latents", "coefficient_ratio", "feature_recall",
]
