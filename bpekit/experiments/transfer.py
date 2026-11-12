"""Domain transfer on the monorepo itself: READMEs (prose) vs Python source vs JSON/data."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from bpekit.transfer import transfer_matrix, transfer_penalty, format_matrix

root = pathlib.Path(__file__).resolve().parents[2]
def gather(pattern, cap=300_000):
    out = []
    for p in sorted(root.glob(pattern)):
        if "bpekit" in p.parts: continue
        out.append(p.read_text(errors="ignore"))
        if sum(map(len, out)) > cap: break
    return "\n".join(out)[:cap]

corp = {"prose": gather("*/README.md"), "python": gather("*/**/*.py"), "json": gather("*/**/*.json")}
corp = {k: v for k, v in corp.items() if len(v) > 5000}
names, m = transfer_matrix(corp, vocab_size=int(sys.argv[1]) if len(sys.argv) > 1 else 600)
print({k: len(v) for k, v in corp.items()})
print(format_matrix(names, m))
print({k: round(v, 3) for k, v in transfer_penalty(names, m).items()})
