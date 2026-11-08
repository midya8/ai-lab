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

## Adapter hub: swap, blend, route, compose

`loralab.hub.AdapterHub` keeps one frozen GPT and any number of named adapters (same targets, ranks may differ):

- **swap**: `hub.active("tech")` loads one adapter; `hub.active(None)` is the bit-exact base model. Weights are restored on exit.
- **blend**: `hub.active({"tech": .5, "kitchen": .5})` uses `W + Σ wᵢ·ΔWᵢ`. It blends the *deltas*, not the A/B factors (`mean(A)·mean(B) ≠ mean(A·B)`, and a test pins that).
- **route**: `hub.route(ids)` scores a short snippet under each adapter (and the base) and returns the lowest-NLL one. No labels, no classifier.
- **compose**: `hub.compose(spec, "mix", rank=k)` folds a blend into a single new adapter with a per-weight truncated SVD, and reports the relative error. It is exact when `k` ≥ the sum of the parts' ranks.
- `hub.save(dir)` / `AdapterHub.load(model, dir)`: one `.npz` per adapter plus `hub.json`.

```
PYTHONPATH=src python -m loralab.hub_demo [--rank 4] [--save DIR]
```

Demo: pretrain on Shakespeare (A), then train a `tech` adapter (B) and a `kitchen` adapter (C) independently from the same frozen base (rank 4, 2,304 params = 3.7% each). Held-out NLL in nats/char, seed 0, ~15 s on CPU. B and C are synthetic template text.

| config | NLL A (Shakespeare) | NLL B (tech) | NLL C (kitchen) |
|---|---|---|---|
| base (no adapter) | 0.368 | 5.690 | 5.497 |
| tech adapter | 3.500 | 1.393 | 3.145 |
| kitchen adapter | 3.537 | 2.995 | 1.444 |
| blend 50/50 | 3.346 | 2.428 | 2.898 |
| composed 50/50 (SVD, rank 8) | 3.346 | 2.428 | 2.898 |

What to take from it, honestly: each specialist wins its own domain and cuts NLL from ~5.6 to ~1.4. Routing picks the right adapter from a 48-character snippet 100% of the time on these two very different grammars (harder, closer domains would be harder). A 50/50 blend is a compromise, not a union: about halfway on both new domains, so blending is for interpolating behaviour, and routing is the better tool when you know which domain you are in. The SVD-composed adapter reproduces the blend to ~1e-7 relative error, so a blend can be shipped as one ordinary adapter file.

Tests (`tests/test_hub.py`, 13): bit-exact base and weight restoration, single adapter equals `GPTLoRA`, linear delta blending, factor-averaging is *not* the blend, exact compose at sum of ranks and monotone truncation error, registry errors, merged-adapter refusal, mixed-rank save/load round-trip, prompt NLL vs manual windows (odd lengths), domain generator, training reduces loss, end-to-end routing/specialisation.

## QLoRA-style: adapters on a quantized base

`loralab.quant` quantizes the frozen GPT's linear weights block-wise (absmax scale per `--block` weights, default 64) in three schemes: `int8`, uniform `int4`, and `nf4` (QLoRA's 4-bit NormalFloat codebook, whose levels sit at normal quantiles and spend the few 4-bit levels where Gaussian-ish weights actually are). `QuantizedBase` swaps the model's `qkv/proj/fc/out` weights for their dequantized values and keeps the integer codes (4-bit codes nibble-packed in `.save()` files), so `GPTLoRA` trains fp32 adapters on top with no other change: the hand-written backward flows through the dequantized weights, which is the QLoRA recipe. `verify_frozen()` proves training never touched the base.

```
PYTHONPATH=src python -m loralab.qlora_demo [--rank 4] [--block 64] [--schemes fp32,int8,nf4,int4]
```

Pretrain on Shakespeare (A) in fp32, quantize, then train rank-4 attention adapters on the synthetic tech domain (B). Held-out NLL in nats/char, seed 0, ~40 s on CPU. The last column trains fresh adapters on A itself over the same quantized base:

| base | model KiB | weight err | NLL A (base) | NLL B (base) | NLL B (+LoRA) | NLL A (+LoRA on A) |
|---|---|---|---|---|---|---|
| fp32 | 241.6 | 0.0000 | 0.368 | 5.690 | 1.422 | 0.249 |
| int8 | 83.0 | 0.0062 | 0.368 | 5.690 | 1.448 | 0.249 |
| nf4 | 56.0 | 0.0940 | 0.373 | 5.621 | 1.442 | 0.253 |
| int4 | 56.0 | 0.1131 | 0.396 | 5.677 | 1.421 | 0.251 |

Reading it honestly: the model is 4.3x smaller (embeddings, biases and LayerNorm stay fp32; the linear weights alone shrink ~3.8x at 8 bit and ~7x at 4 bit, scales included). int8 is lossless in practice (0.6% weight error). At 4 bits the plain base degrades on A (int4 0.368 → 0.396) and NF4 loses less than uniform int4 at the same size (0.373), matching the codebook's design goal. LoRA on the quantized base reaches the fp32 adapter's domain-B NLL within noise (all four land in 1.42-1.45; one seed, so differences of a few hundredths are not meaningful). Training adapters on A closes the quantization gap completely, but note that fp32 improves too (0.368 → 0.249): that column shows the adapters absorbing the precision loss, not a recovery beyond the fp32 baseline. A 62k-parameter model is a forgiving test: its weights are over-trained on a small corpus, and bigger models tolerate 4 bits better still.

Tests (`tests/test_quant.py`, 20): NF4 codebook shape, int8 half-step error bound, error ordering int8 < nf4 < int4 on Gaussian weights, block-size monotonicity, an outlier only damaging its own block, zero blocks and padding, exact reproduction of codebook levels, nibble pack/unpack, stored-byte ratios, only targeted tensors change, parameters equal dequantized codes, finite-difference adapter gradients through an int4 base, frozen codes after training, save/load round trip for every scheme, demo smoke test.
