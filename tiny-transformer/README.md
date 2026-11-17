# tiny-transformer

A GPT-style decoder in plain NumPy, backprop included. No autograd and no PyTorch. Every op's backward pass is checked against finite differences. Over time I added most of the things modern LLMs use so I could see each one work at a size I can actually reason about.

```bash
cd tiny-transformer
pip install -r requirements.txt
PYTHONPATH=src python -m tinygpt.train --data data/corpus.txt --steps 300
PYTHONPATH=src python -m tinygpt.cli generate tinygpt.npz --prompt "To be" -n 200
PYTHONPATH=src python -m pytest -q          # 147 tests
```

A 108k-parameter model goes from loss 3.63 to ~0.25 on the bundled corpus in about 30 s on a CPU and starts reproducing the text. The corpus is tiny and repetitive, so that's mostly memorisation.

What's in it: pre-LayerNorm blocks, multi-head, grouped-query and multi-query attention, learned positions or RoPE with context extension (position interpolation and NTK scaling), KV-cache decoding, a byte-level BPE tokenizer, AdamW with warmup and cosine decay, gradient accumulation and clipping, dropout, weight tying, label smoothing, early stopping, an LR finder, top-k/top-p sampling with a repetition penalty, and speculative decoding.

## Experiments

- **RoPE vs learned positions:** about the same inside the training window. RoPE can run past it but loss climbs quickly (0.22 at 32 tokens, 1.22 at 96).
- **Context extension:** zero-shot, NTK scaling roughly halves the damage at 2x length (0.96 vs 1.44 nats). Unfine-tuned linear interpolation is the worst option. After fine-tuning, everything refits the memorised corpus, so that comparison doesn't tell you much.
- **GQA/MQA:** the cache savings are exact (8 KV heads down to 1 is 8x smaller). Quality differences between variants were within noise across three runs.
- **Key-value recall:** with 2 pairs, RoPE reliably learns it (0.98) while learned positions depend on the seed (0.79 mean, 0.58 worst). With 4 pairs neither converged in the budget.

All of these are one to three seeds on a very small corpus, so read them as directions. [`microgrid`](../microgrid) reproduces this model's gradients with tape autodiff, and [`lora`](../lora) trains adapters on it.
