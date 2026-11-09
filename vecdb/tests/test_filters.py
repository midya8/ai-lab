import pytest

from vecdb import compile_filter

DOC = {"lang": "en", "year": 2022, "tags": "a", "score": 0.5}


@pytest.mark.parametrize("spec,expected", [
    ({"lang": "en"}, True),
    ({"lang": "fr"}, False),
    ({"year": {"$gte": 2020, "$lt": 2023}}, True),
    ({"year": {"$gt": 2022}}, False),
    ({"year": {"$ne": 2021}}, True),
    ({"tags": {"$in": ["a", "b"]}}, True),
    ({"tags": {"$nin": ["a"]}}, False),
    ({"lang": "en", "year": 2021}, False),                   # implicit AND
    ({"$or": [{"lang": "fr"}, {"year": 2022}]}, True),
    ({"$and": [{"lang": "en"}, {"score": {"$lt": 0.4}}]}, False),
    ({"$not": {"lang": "fr"}}, True),
    ({"missing": "x"}, False),
    ({"missing": {"$gt": 1}}, False),
    ({"missing": {"$ne": 1}}, True),
    ({"missing": {"$nin": [1]}}, True),
    ({"lang": {"$gt": 5}}, False),                            # str vs int: no match, no crash
    ({}, True),
])
def test_matching(spec, expected):
    assert compile_filter(spec)(DOC) is expected


@pytest.mark.parametrize("bad", [[], {"$xor": []}, {"a": {"$bogus": 1}}, {"$and": []}, {"a": {"$in": 5}}])
def test_invalid_specs_fail_at_compile_time(bad):
    with pytest.raises(ValueError):
        compile_filter(bad)
