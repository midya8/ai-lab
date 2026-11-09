# Temperature scaling before confident learning

`python experiments/calibration_effect.py` (blobs, 20% symmetric noise, 5-fold, 3 seeds).

| model | seed | T | ECE raw->cal | est. noise raw | est. noise cal | true | F1 raw | F1 cal |
|---|---|---|---|---|---|---|---|---|
| softmax l2=1e-2 | 0 | 0.98 | 0.123->0.118 | 0.234 | 0.234 | 0.200 | 0.858 | 0.859 |
| softmax l2=1e-2 | 1 | 0.98 | 0.096->0.097 | 0.248 | 0.248 | 0.200 | 0.812 | 0.809 |
| softmax l2=1e-2 | 2 | 0.98 | 0.124->0.120 | 0.223 | 0.223 | 0.200 | 0.905 | 0.905 |
| softmax l2=5 | 0 | 0.07 | 0.397->0.083 | 0.231 | 0.238 | 0.200 | 0.788 | 0.854 |
| softmax l2=5 | 1 | 0.07 | 0.330->0.051 | 0.283 | 0.282 | 0.200 | 0.781 | 0.744 |
| softmax l2=5 | 2 | 0.07 | 0.523->0.139 | 0.224 | 0.218 | 0.200 | 0.823 | 0.904 |
| kNN k=15 | 0 | 0.88 | 0.125->0.103 | 0.237 | 0.238 | 0.200 | 0.859 | 0.843 |
| kNN k=15 | 1 | 0.91 | 0.083->0.070 | 0.261 | 0.261 | 0.200 | 0.765 | 0.761 |
| kNN k=15 | 2 | 0.90 | 0.110->0.093 | 0.230 | 0.231 | 0.200 | 0.892 | 0.885 |

**Reading:** a well-calibrated model (default softmax, kNN) gets T near 1 and nothing changes. A heavily over-regularised softmax (l2=5) is very under-confident (ECE 0.33-0.52); calibration fixes ECE (to 0.05-0.14) and F1 moves +0.07 and +0.08 on two seeds but -0.04 on one. Only 3 seeds, so treat that as a hint, not a result. The noise-rate estimate barely moves, so calibration does not cure the over-estimate noted in v0.1. T hit 0.07, close to the 0.05 search floor.
