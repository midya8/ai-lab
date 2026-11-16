"""python -m gplab {hyper,sparse,features,calibration,extrapolation,scaling}"""
from __future__ import annotations

import argparse
import json

from . import studies as S

STUDIES = {
    "hyper": S.hyperparameter_recovery,
    "sparse": S.sparse_methods,
    "features": S.feature_approximations,
    "calibration": S.calibration,
    "extrapolation": S.extrapolation,
    "scaling": S.scaling,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="gplab", description=__doc__)
    p.add_argument("study", choices=sorted(STUDIES))
    a = p.parse_args(argv)
    print(json.dumps(STUDIES[a.study](), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
