# lora-lab

LoRA (low-rank adaptation) implemented from scratch in NumPy with manual backprop.

- `LoRALinear`: frozen `W` plus trainable low-rank `A·B` (B zero-init), scaled by `alpha/r`; merge/unmerge for zero-overhead inference.
- A 2-layer MLP is pretrained on task A, then adapted to a shifted task B training only the adapters.
- Tests: zero-init identity, finite-difference gradient check, merge equivalence, frozen base weights, adaptation beats base.

```
pip install -r requirements.txt
PYTHONPATH=src python -m loralab.cli --rank 2
python -m pytest
```

## Rank sweep

`PYTHONPATH=src python -m loralab.cli --sweep` adapts the same pretrained MLP at ranks 1/2/4/8/16 and a full fine-tune reference, printing trainable params, % of full, and task-B accuracy.
On this synthetic Gaussian-blob task every setting reaches 1.000, so the sweep shows the parameter savings (rank 1 trains 6.6% of the weights) rather than an accuracy gap; the task is too easy to separate ranks. Numbers are synthetic.
