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
