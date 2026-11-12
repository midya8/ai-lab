"""python experiments/sweep.py FILE  (run with PYTHONPATH=src)"""
import sys
from bpekit.sweep import sweep, format_table
text = open(sys.argv[1], encoding="utf-8").read()
print(format_table(sweep(text, [300, 400, 600, 800, 1200, 1600])))
