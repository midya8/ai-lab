# mamba-numpy

A Mamba-style selective state-space model in NumPy with hand-written backprop through the scan. The backward pass is itself a reverse-time scan, and every gradient is checked against finite differences, including a test that corrupts the SSM backward on purpose to make sure the check catches it.

I wanted one experiment that isolates what "selective" actually buys, so the main comparison is two models that differ only in whether the step size and the B/C matrices depend on the input.

```bash
cd mamba-numpy
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                                                    # 37 tests
PYTHONPATH=src python -m mamba train --steps 1500 --seq-len 16 --save /tmp/sel.npz    # ~35 s
PYTHONPATH=src python -m mamba train --lti --steps 1500 --seq-len 16                  # fixed-dynamics baseline
PYTHONPATH=src python -m mamba eval /tmp/sel.npz --seq-len 32                         # longer than training
```

## Selective copying

Four symbols are scattered among L noise tokens and have to be repeated in order at the end. Two-layer, d_model 16 models, 3 seeds each. "2L" is the same model tested on noise twice as long as it ever saw.

| noise L | model | token acc | all 4 right | all 4 right at 2L |
|--:|--|--:|--:|--:|
| 16 | selective | 1.000 | 1.000 | 0.962 |
| 16 | fixed (LTI) | 0.909 | 0.680 | 0.110 |
| 32 | selective | 1.000 | 0.999 | 0.972 |
| 32 | fixed (LTI) | 0.867 | 0.536 | 0.077 |
| 64 | selective | 1.000 | 0.999 | 0.996 |
| 64 | fixed (LTI) | 0.810 | 0.389 | 0.056 |

The selective model solves it and generalises to longer sequences. The fixed one gets worse as noise grows and falls apart off-distribution, since a fixed convolution can't learn to skip the noise. Widening the LTI model to slightly more parameters than the selective one barely helps (0.654 exact at L=32), so it isn't a size thing.

## Other notes

- Recurrent decoding with a fixed-size state matches the full forward pass to 1e-10. The state is 77,824 bytes per sequence at any length. A KV cache of the same shape would be 33.5 MB at 8192 tokens (that's arithmetic, not a measured transformer).
- The parallel associative scan is exact but about 9x slower than a plain loop in NumPy. It only pays off on parallel hardware, so the sequential scan is the default.
- I probed what the step size learns. It's higher on data tokens than noise in 5 of 6 layers, and zeroing it on data tokens hurts two of three seeds badly. The third seed routes the information some other way. So "delta gates the input" is only partly the story.

Tiny models (~7k params) on synthetic tasks. Associative recall is implemented but I haven't run a study on it yet.
