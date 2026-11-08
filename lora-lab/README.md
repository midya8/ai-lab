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

## LoRA on the tiny-transformer GPT (attention adapters)

`loralab.gpt_lora.GPTLoRA` wraps the NumPy GPT from [`tiny-transformer`](../tiny-transformer) and trains low-rank adapters on its attention matrices (`qkv`, `proj`; optionally the MLP's `fc`, `out`) while every base weight stays frozen.

No new backward pass was needed: `tinygpt` already backpropagates by hand to `dL/dW_eff`, and with `W_eff = W + s·A·B` (`s = alpha/r`) the chain rule gives `dA = s·dW·Bᵀ` and `dB = s·Aᵀ·dW`. Each step swaps `W_eff` in, reuses that backward, restores `W`, and keeps only the adapter gradients. Features: zero-init identity, merge/unmerge into the base weights, adapter on/off toggle (off = bit-exact base model), small `.npz` adapter files with rank/target validation.

```
PYTHONPATH=src python -m loralab.gpt_demo --rank 4                      # attention only (qkv, proj)
PYTHONPATH=src python -m loralab.gpt_demo --rank 4 --targets qkv,proj,fc,out
```

Demo: pretrain a 2-layer char GPT (61,862 params) on Shakespeare, then adapt it to a template-generated "technical prose" domain it has never seen. Held-out NLL in nats/char, seed 0, ~20 s on CPU (synthetic data; domain-A numbers are low because the small corpus repeats, so read them as relative):

| method | NLL A (Shakespeare) | NLL B (tech prose) | trainable |
|---|---|---|---|
| pretrained, no adapter | 0.368 | 5.690 | 0 |
| LoRA r=1, qkv+proj | 2.721 | 2.122 | 576 (0.9%) |
| LoRA r=4, qkv+proj | 3.234 | 1.504 | 2,304 (3.7%) |
| LoRA r=8, qkv+proj | 3.860 | 1.036 | 4,608 (7.4%) |
| LoRA r=4, qkv+proj+fc+out | 4.044 | 0.916 | 6,144 (9.9%) |
| full fine-tune | 5.466 | 0.381 | 61,862 (100%) |

Reading it honestly: full fine-tuning fits the new domain best but wipes out the old one (A: 0.37 → 5.47) and needs a whole new checkpoint. LoRA closes most of the gap on B with a few percent of the weights and forgets far less; "forgets less" is measured with the adapter on, and switching it off returns the exact base model (A: 0.368), which is the practical win: one frozen base, many tiny swappable adapters. More rank/targets buy a better fit on B at the cost of more drift on A.

Tests (`tests/test_gpt_lora.py`, 10): zero-init equals base logits, finite-difference gradient check on every adapter incl. MLP targets, base weights untouched after training, loss decreases, merge ≡ adapted forward and unmerge restores, adapter-off is bit-exact base, parameter counts, adapter save/load round-trip with mismatch rejection, deterministic domain text, end-to-end adaptation.
