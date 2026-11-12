"""datadoctor: find label errors, duplicates, leakage and outliers in a tabular dataset."""

__version__ = "0.1.0"

from .confident import (  # noqa: F401
    calibrate_joint,
    class_thresholds,
    confident_joint,
    estimate_noise_matrix,
    find_label_issues,
    per_class_summary,
    suggest_relabels,
    rank_label_issues,
)
from .cv import cross_val_probs, stratified_kfold  # noqa: F401
from .models import SoftmaxRegression  # noqa: F401
from .calibration import apply_temperature, calibrate, fit_temperature  # noqa: F401
from .multilabel import (  # noqa: F401
    cross_val_probs_multilabel,
    cooccurrence_refine,
    make_cooccurring_labels,
    find_multilabel_issues,
    inject_multilabel_noise,
    rank_multilabel_issues,
)
from .balance import class_balance, label_shift  # noqa: F401
from .drift import feature_drift, ks_statistic, psi  # noqa: F401
from .health import feature_health  # noqa: F401
