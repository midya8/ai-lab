import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from triage.dedupe import find_duplicates, cluster

T = [
    "I was charged twice for my subscription",
    "I was charged twice for my subscription!!",
    "Charged twice for my subscription, please help",
    "The app crashes when I open the dashboard",
    "Please add dark mode",
]


def test_pairs_found_and_sorted():
    p = find_duplicates(T, 0.7)
    assert (0, 1) == p[0][:2]
    assert all(i < j for i, j, _ in p)
    assert [s for *_, s in p] == sorted((s for *_, s in p), reverse=True)


def test_cluster_groups_transitively():
    assert cluster(T, 0.6) == [[0, 1, 2]]


def test_unrelated_not_clustered():
    assert cluster(T[3:], 0.6) == []


def test_empty_and_threshold_one():
    assert cluster([]) == []
    assert find_duplicates(["a b", "a c"], 1.0) == []
