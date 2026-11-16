import json
import sys

from . import studies as S

USAGE = "usage: python -m hoplab {capacity|stress|basin|correlated|beta|glyphs} [--json]"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"capacity", "stress", "basin", "correlated", "beta", "glyphs"}:
        print(USAGE)
        return 2
    as_json = "--json" in argv
    cmd = argv[0]
    if cmd == "glyphs":
        rows = S.glyph_study()
    elif cmd == "stress":
        rows = S.capacity_study(n=100, ps=(100, 400, 1000, 2000, 4000), noise=0.25, queries=20,
                                rules=("dam-poly3", "dam-poly5", "dam-exp"))
    else:
        rows = {"capacity": S.capacity_study, "basin": S.basin_study,
                "correlated": S.correlated_study, "beta": S.beta_study}[cmd]()
    print(json.dumps(rows, indent=2) if as_json else S.fmt_table(rows))
    if cmd == "glyphs" and not as_json:
        print(S.glyph_example())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
