import math

import numpy as np

from codesearch.chunker import Chunk
from codesearch.index import DEFAULT_WEIGHTS, Index


def mk(qual, doc="", ids=(), path="m.py"):
    return Chunk(path, qual, "function", 1, 3, signature=f"def {qual}()", doc=doc, ids=list(ids))


def test_name_match_outweighs_body_match():
    chunks = [mk("alpha", ids=["gamma"] * 3), mk("gamma"), mk("filler_one", ids=["x"]), mk("filler_two", ids=["y"])]
    ix = Index.build(chunks)
    s = ix.bm25f(["gamma"])
    assert s[1] > s[0] > 0 and s[2] == 0


def test_rare_terms_outweigh_common_ones():
    chunks = [mk("a_common", ids=["common", "rareterm"]), mk("b", ids=["common"]), mk("c", ids=["common"]), mk("d", ids=["common"])]
    ix = Index.build(chunks)
    assert ix.idf["rareterm"] > ix.idf["common"]
    assert np.argmax(ix.bm25f(["rareterm", "common"])) == 0


def test_term_frequency_saturates():
    ix = Index.build([mk("a", ids=["tok"] * 2), mk("b", ids=["tok"] * 40), mk("c", ids=["other"])])
    s = ix.bm25f(["tok"])
    assert s[1] < 2.2 * s[0]


def test_unseen_term_scores_zero_everywhere():
    ix = Index.build([mk("a"), mk("b")])
    assert not ix.bm25f(["nothing"]).any()


def test_matches_a_hand_computed_score_for_one_field():
    w = {f: 0.0 for f in DEFAULT_WEIGHTS}
    w["doc"] = 1.0
    chunks = [mk("a", doc="needle hay"), mk("b", doc="hay hay"), mk("c", doc="hay")]
    ix = Index.build(chunks, weights=w, b={f: 0.0 for f in DEFAULT_WEIGHTS}, k1=1.2)
    idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert math.isclose(ix.bm25f(["needl"])[0] if "needl" in ix.idf else ix.bm25f(["needle"])[0],
                        idf * 1.0 / (1.2 + 1.0), rel_tol=1e-9)


def test_flat_bm25_and_name_only_baselines():
    ix = Index.build([mk("parse_diff"), mk("other", doc="parse the diff")])
    assert ix.name_only(["pars", "diff"]).tolist() == [2.0, 0.0]
    assert ix.bm25_flat(["pars"]).max() > 0
