"""datadoctor: find label errors, duplicates, leakage and outliers in a tabular dataset."""

__version__ = "0.1.0"

from .confident import (  # noqa: F401
    calibrate_joint,
    class_thresholds,
    confident_joint,
    estimate_noise_matrix,
    find_label_issues,
    rank_label_issues,
)
from .cv import cross_val_probs, stratified_kfold  # noqa: F401
from .models import SoftmaxRegression  # noqa: F401
from .calibration import apply_temperature, calibrate, fit_temperature  # noqa: F401
