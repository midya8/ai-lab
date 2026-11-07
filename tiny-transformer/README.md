# tiny-transformer (tinygpt)

A GPT-style decoder-only transformer written in **pure NumPy, including hand-derived backpropagation**. No autograd, no PyTorch.

- Pre-LayerNorm blocks, causal multi-head attention, GELU MLP, learned positional embeddings
- Manual backward pass for every op, verified against finite differences in the test suite
- Adam with gradient clipping, char tokenizer, temperature / top-k sampling, save/load

## Run
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m tinygpt.train --data data/corpus.txt --steps 300
python -m pytest
```
A 108k-parameter model drops from loss 3.63 to ~0.25 on the bundled corpus in about 30s on a CPU and starts producing the text.

## KV-cache decoding
`tinygpt.generate_cached(model, prompt, n)` prefills per-layer keys/values once and decodes each token in O(T) instead of re-running the full sequence. It is verified to produce token-identical output to `GPT.generate` (greedy and top-k sampling), including the sliding-window case past `block_size`.

## BPE tokenizer
`tinygpt.BPETokenizer.train(text, vocab_size)` learns byte-level BPE merges from scratch (256 byte base vocab, lossless on any Unicode, JSON save/load). Same interface as `CharTokenizer`, so it plugs into the trainer's data pipeline.

## Tests
Gradient check on every parameter tensor, causality (future tokens never affect past logits), initial loss ≈ ln(vocab), overfitting a tiny corpus and generating it, checkpoint round trip.

## Training upgrades (v0.4)
`--warmup N` and cosine decay to 10% of peak LR (`tinygpt.warmup_cosine`), decoupled AdamW weight decay on weight matrices (`--weight-decay`), and `--tokenizer bpe --vocab 300` to train on the from-scratch BPE tokenizer.

## Self-contained checkpoints + perplexity (v0.5)
`save_checkpoint` / `load_checkpoint` bundle weights, model config and tokenizer (char or BPE) in one `.npz`, so a model can be reloaded with no side files. `evaluate` computes deterministic held-out loss and perplexity; `train` now holds out the tail (`--val-frac`, default 0.1) and reports it.
```bash
PYTHONPATH=src python -m tinygpt.cli generate tinygpt.npz --prompt "To be" -n 200
PYTHONPATH=src python -m tinygpt.cli eval tinygpt.npz --data data/corpus.txt --tail 0.1
```
