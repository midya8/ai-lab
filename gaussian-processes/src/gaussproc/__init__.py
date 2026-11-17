"""gaussproc: Gaussian processes from scratch in NumPy."""
from .data import sample_gp_data, seasonal_trend, smooth_function
from .features import FeatureGP, NystromFeatures, RandomFourierFeatures, kernel_approx_error
from .classify import LaplaceGPC, expected_calibration_error, log_loss
from .gp import ExactGP, coverage, gaussian_nlpd
from .kernels import RBF, Kernel, Linear, Matern32, Matern52, Periodic, Product, SpectralMixture, Sum
from .hetero import HeteroGP
from .kalman import KalmanMatern
from .multioutput import ICMGP
from .optim import lbfgs
from .robust import RobustGP
from .svgp import SVGP
from .sparse import SparseGP, greedy_inducing, kmeans_inducing

__all__ = [
    "LaplaceGPC", "log_loss", "expected_calibration_error",
    "RobustGP", "HeteroGP", "ICMGP", "KalmanMatern", "ExactGP", "SparseGP", "SVGP", "FeatureGP", "RandomFourierFeatures", "NystromFeatures",
    "RBF", "Matern32", "Matern52", "Periodic", "SpectralMixture", "Linear", "Sum", "Product", "Kernel",
    "greedy_inducing", "kmeans_inducing", "kernel_approx_error", "lbfgs",
    "gaussian_nlpd", "coverage", "sample_gp_data", "seasonal_trend", "smooth_function",
]
__version__ = "0.8.0"
