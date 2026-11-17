#!/bin/sh
# Reproduces experiments/results_seeds_0_1_2.txt (about 2 minutes per seed; run them in parallel if you like).
cd "$(dirname "$0")/.." || exit 1
for s in 0 1 2; do
  echo "=== seed $s"
  PYTHONPATH=src python -m mechinterp --seed "$s" --steps 4000 | grep -v '^step'
done
