# tiny-transformer (tinygpt)

A GPT-style decoder-only transformer written in **pure NumPy, including hand-derived backpropagation**. No autograd, no PyTorch.

- Pre-LayerNorm blocks, causal multi-head / grouped-query / multi-query attention (`--kv-heads`), GELU MLP, learned positional embeddings, or rotary embeddings (`--rope`)
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

## Label smoothing

`--label-smoothing EPS` trains against `(1-EPS)*onehot + EPS/V` targets (loss = `(1-EPS)*NLL + EPS*mean(-log p)`, gradient `p - q`, finite-difference-checked). Only the training loss/gradients change; validation loss and perplexity stay plain NLL so runs remain comparable.

## Per-position loss (context usage)

`python -m tinygpt.cli eval ckpt.npz --data corpus.txt --by-position 4` prints mean loss per
context-position bin and the "context gain" (first bin minus last bin). A model that uses its
context shows loss falling with position; `evaluate_by_position` exposes the full per-position curve.

## Sliding-window perplexity

`eval ckpt.npz --data corpus.txt --stride 4` also prints sliding-window perplexity: windows of
`block_size` advance by `stride` and only newly covered tokens are scored, so each token gets as much
context as possible. `--stride block_size` equals plain eval; smaller strides lower perplexity.

## Rotary position embeddings (RoPE)

`--rope` (`GPTConfig.rope`) replaces the learned absolute position table with rotary embeddings: each head's q and k are rotated, pair-wise across the two halves of the head dimension, by an angle proportional to the token position, so attention scores depend only on relative distance. There is no `wpe` parameter (fewer params), the backward pass applies the inverse rotation to `dq`/`dk`, and the KV cache stores keys already rotated for their own position. Requires an even head dimension. Verified in `tests/test_rope.py`: finite-difference gradients (also with tied weights and dropout masks), norm preservation, relative-position-only scores, causality, KV-cache parity past `block_size`, checkpoint round trip, and old configs still loading as learned positions.

```bash
PYTHONPATH=src python -m tinygpt.train --rope --block 32 --steps 300
PYTHONPATH=src python experiments/rope_vs_learned.py     # same seed/hparams, positions differ
```

Held-out loss (nats/char, trained with block 32, 2.6k-char corpus; one seed, small numbers, treat as indicative only):

| positions | params | len 16 | len 32 | len 64 | len 96 |
|---|---|---|---|---|---|
| learned | 106,917 | 0.392 | 0.235 | n/a | n/a |
| rope | 104,869 | 0.376 | 0.217 | 0.508 | 1.223 |

RoPE matches learned positions inside the training window (a second run, seed 1 / 500 steps, gave 0.340/0.197 vs 0.339/0.196). It *can* be run past `block_size` (a learned table cannot), but at this scale it does **not** extrapolate gracefully: loss climbs on unseen lengths. The script reports that rather than hiding it; length extrapolation usually needs position interpolation or training on varied lengths.

## Grouped-query attention (GQA / MQA)

`--kv-heads N` (`GPTConfig.n_kv_head`) gives the model `N` key/value heads shared by groups of `n_head / N` query heads: `N = n_head` is ordinary multi-head attention (the default, identical weights layout as before), `1 < N < n_head` is GQA, `N = 1` is multi-query attention. The `qkv` projection shrinks to `C + 2*N*head_dim` columns. Forward repeats each kv head for its query group (`expand_kv`); backward sums the gradients of every query head in a group back onto the shared head (`reduce_kv`, the exact adjoint, tested). The KV cache stores only the `N` compact heads, so inference memory falls by `n_head / N` (`kv_cache_bytes(cfg, seq_len)`). Works with RoPE, tied weights, dropout and checkpoints; old checkpoints without the field load as MHA. Verified in `tests/test_gqa.py`: finite-difference gradients (GQA and MQA, with RoPE, tying and dropout masks), MQA exactly equals an MHA model whose K/V head weights are tied copies, causality, cached decoding equals full forward and `generate`, compact cache shapes, invalid group sizes rejected.

```bash
PYTHONPATH=src python -m tinygpt.train --heads 8 --kv-heads 2 --steps 300
PYTHONPATH=src python experiments/gqa_vs_mha.py          # MHA vs GQA vs MQA, same seed/hparams
```

Held-out loss (nats/char, 8 query heads, 2.6k-char corpus, block 32; the validation tail is from a small repetitive corpus, so absolute losses are low):

| kv heads | variant | params | KV cache | seed 0 (300 steps) | seed 1 (300) | seed 2 (500) |
|---|---|---|---|---|---|---|
| 8 | MHA | 106,917 | 32.0 KiB (1x) | 0.202 | 0.178 | 0.164 |
| 4 | GQA | 98,597 | 16.0 KiB (0.5x) | 0.261 | 0.212 | 0.183 |
| 2 | GQA | 94,437 | 8.0 KiB (0.25x) | 0.261 | 0.209 | 0.148 |
| 1 | MQA | 92,357 | 4.0 KiB (0.125x) | 0.239 | 0.248 | 0.145 |

Honest reading: the cache and parameter savings are exact. Quality is mixed: at 300 steps MHA was ahead by 0.03-0.08 nats on both seeds, while at 500 steps GQA/MQA matched or beat it (seed 2). Three single-seed runs on a tiny corpus cannot rank the variants; the order between 4, 2 and 1 kv heads is noise. What the evidence does support is that cutting the cache 4-8x costs little at this scale, which is the usual motivation for GQA.
