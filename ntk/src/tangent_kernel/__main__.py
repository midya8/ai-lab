"""python -m tangent_kernel {convergence,lazy,ensemble,spectral,alpha,depth} [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from . import studies as S

STUDIES = {
    "convergence": S.kernel_convergence,
    "lazy": S.lazy_vs_width,
    "ensemble": S.ensemble_vs_analytic,
    "spectral": S.spectral_bias,
    "alpha": S.lazy_alpha,
    "depth": S.depth_spectrum,
}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tangent_kernel", description=__doc__)
    p.add_argument("study", choices=sorted(STUDIES))
    p.add_argument("--json", action="store_true", help="print raw JSON")
    a = p.parse_args(argv)
    out = STUDIES[a.study]()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
