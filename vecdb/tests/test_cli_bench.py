import subprocess
import sys
import os

from vecdb.bench import make_data, run_bench, to_markdown

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_run_bench_small_is_consistent():
    rows = run_bench("clusters", n=1500, dim=16, nq=20, k=5, pq_m=4, hnsw_efc=40, log=lambda s: None)
    by = {(r["method"].split()[0], r["setting"]): r for r in rows}
    assert rows[0]["recall"] == 1.0
    ivf = [r for r in rows if r["method"].startswith("IVF-Flat")]
    assert [r["recall"] for r in ivf] == sorted(r["recall"] for r in ivf)
    assert ivf[0]["scanned"] < ivf[-1]["scanned"] < 1500
    adc = next(r for r in rows if "ADC only" in r["setting"])
    flat = rows[0]
    assert adc["bytes"] < flat["bytes"] / 2   # codes + 8-byte ids + codebooks, at tiny n
    hn = [r for r in rows if r["method"].startswith("HNSW")]
    assert hn[-1]["recall"] >= hn[0]["recall"] and hn[-1]["recall"] > 0.8
    md = to_markdown(rows)
    assert md.count("\n") == len(rows) + 1


def test_make_data_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        make_data("nope", 10, 4, 2)


def test_cli_demo_runs():
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src"))
    out = subprocess.run([sys.executable, "-m", "vecdb", "demo", "--n", "400"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "unfiltered top 3" in out.stdout and "filtered" in out.stdout
