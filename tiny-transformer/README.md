# tiny-transformer (tinygpt)

A GPT-style decoder-only transformer written in **pure NumPy, including hand-derived backpropagation**. No autograd, no PyTorch.

- Pre-LayerNorm blocks, causal multi-head attention, GELU MLP, learned positional embeddings
- Manual backward pass for every op, verified against finite differences in the test suite
- LR range test (`--find-lr`), Adam with gradient clipping, char tokenizer, temperature / top-k / top-p (nucleus) sampling (`--top-p`), CTRL-style repetition penalty (`--repetition-penalty`), save/load

## Run
```bash
pip install -r requirements.txt
PYTHONPATH=src python -m tinygpt.train --data data/corpus.txt --steps 300
python -m pytest
```
A 108k-parameter model drops from loss 3.63 to ~0.25 on the bundled corpus in about 30s on a CPU and starts producing the text.

## KV-cache decoding
`tinygpt.generate_cached(model, prompt, n)` prefills per-layer keys/values once and decodes each token in O(T) instead of re-running the full sequence. It is verified to produce token-identical output to `GPT.generate` (greedy, top-k and top-p sampling), including the sliding-window case past `block_size`.

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

## Gradient accumulation

`--micro-batch N` splits each batch into micro-batches of N rows and averages their gradients (`tinygpt/accum.py`), so large effective batches fit in less memory. Tests verify the result equals the full-batch gradient.

## Early stopping

`--patience N` evaluates on the held-out tail every `--eval-every` steps (default 50) and stops after N evaluations without a `--min-delta` improvement, then restores the best weights before saving (`tinygpt/earlystop.py`).

## Gradient clipping

`--clip X` rescales all gradients so their global L2 norm is at most X (`tinygpt/clip.py`, same semantics as `clip_grad_norm_`) and logs the pre-clip norm. Applied after accumulation, before the optimizer step.

## Dropout

`--dropout P` applies inverted dropout to the embeddings, attention probabilities, and both residual branches of every block, with hand-written backward passes through the saved masks (verified against finite differences with fixed masks). It is active only when a training RNG is passed to `forward`/`loss_and_grads`; evaluation, generation, and the KV-cache path are deterministic. The rate is stored in the checkpoint config; older checkpoints load with dropout 0.

## Weight tying

`--tie-weights` shares the token embedding with the output head (`logits = h @ wte.T`), removing `vocab × n_embd` parameters. The embedding gradient receives both the lookup and output-head contributions; verified against finite differences in `tests/test_tying.py`. Stored in the checkpoint config; older checkpoints load as untied.

## LR finder

`python -m tinygpt.train --find-lr --steps 80` runs a Smith-style learning-rate range test (exponential sweep 1e-6 to 1, bias-corrected smoothed loss, stops on divergence), prints the LR at the steepest descent and exits. Model weights are restored, so it has no side effects (`tinygpt/lrfind.py`). Treat the suggestion as a starting point.
