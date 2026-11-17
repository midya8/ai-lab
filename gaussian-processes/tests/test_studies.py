import numpy as np

import gaussproc.studies as S
from gaussproc.__main__ import main


def test_scaling_study_shapes_and_memory_model():
    r = S.scaling(ns=(100, 200), M=10, D=20, reps=1)
    assert [row["n"] for row in r["rows"]] == [100, 200]
    row = r["rows"][1]
    assert row["exact_bytes"] == 8 * 200 * 200 and row["vfe_bytes"] == 8 * 200 * 10


def test_feature_study_runs_and_error_decreases():
    r = S.feature_approximations(n=60, ds=(10, 40), seeds=1)
    assert r["rows"][1]["rff"]["kernel_err"] < r["rows"][0]["rff"]["kernel_err"]
    assert r["rows"][1]["nystrom_greedy"]["kernel_err"] < r["rows"][0]["nystrom_greedy"]["kernel_err"]


def test_hyper_study_small():
    r = S.hyperparameter_recovery(ns=(40,), seeds=2, restarts=1)
    row = r["rows"][0]
    assert 0 < row["lengthscale"]["median"] < 5 and 0 <= row["restarts_in_best_basin"] <= 1


def test_cli_prints_json(capsys, monkeypatch):
    import gaussproc.__main__ as M
    monkeypatch.setitem(M.STUDIES, "scaling", lambda: {"ok": True})
    assert main(["scaling"]) == 0
    assert '"ok": true' in capsys.readouterr().out
