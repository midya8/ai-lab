import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))
from curve import main

def test_heldout_compression_monotone_and_below_train():
    rows = main(sizes=(256, 300, 400))
    assert rows[0][2] == 1.0
    assert rows[0][2] < rows[1][2] < rows[2][2]
    assert all(h <= t for _, t, h in rows)
