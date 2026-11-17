import json
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.dirname(__file__))


def test_cli_depth_emits_json():
    out = subprocess.run([sys.executable, "-m", "tangent_kernel", "depth"], cwd=HERE, capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.path.join(HERE, "src")}, check=True).stdout
    d = json.loads(out)
    assert [r["depth"] for r in d["rows"]] == [1, 2, 3, 4, 6, 8]


def test_cli_rejects_unknown_study():
    r = subprocess.run([sys.executable, "-m", "tangent_kernel", "nope"], cwd=HERE, capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(HERE, "src")})
    assert r.returncode != 0
