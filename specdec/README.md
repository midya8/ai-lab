# specdec

Speculative sampling in NumPy (Leviathan et al. 2023, Chen et al. 2023). A cheap draft model proposes k tokens, the target verifies them, and the output distribution is exactly the target's. Tests check that with chi-square on the joint distribution of the first tokens.

```bash
cd specdec
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python experiment.py
PYTHONPATH=src python experiment_neural.py 300       # real (tiny) transformer pair
PYTHONPATH=src python -m specdec FILE                # calibrate and recommend k for a byte n-gram pair
```

Most experiments use n-gram models where I can dial how close the draft is to the target. Later ones use a pair of small transformers trained with [`microgrid`](../microgrid).

## What I found

The basic result is what you'd hope. With a good draft (95% mixed toward the target) and k=8 you get 8.2 tokens per target call. With a weak draft, larger k actually hurts.

- **Adaptive k** (picking k each round from recent acceptance) stays within ~3% of the best fixed k without knowing draft quality up front.
- **Draft trees** look great on tokens per target call, but once you charge for the extra draft nodes, a single chain (m=1) wins almost everywhere. Two candidates per level only paid off with a poor draft on a peaky target, by up to ~6%.
- **Real transformers:** the draft has 0.19x the parameters but runs at 0.49x the target's time (NumPy overhead dominates tiny nets). The best case is ~15% cheaper per token at k=2, and k >= 4 is worse than plain sampling. You need a draft that's much cheaper, not just smaller.
- **Prompt lookup** as a model-free draft is great on repetitive text (4.16 tokens per call) and useless on text without repeats.
- **Batched verification** keeps the output bit-identical but gave no wall-clock win at this scale, with no KV cache and NumPy overhead.

## Caveats

Most "speedup" numbers are modelled from a cost ratio, not measured wall-clock. Seeds are few with no error bars, and the neural models are undertrained. The empirical acceptance rate undercounts slightly, because proposals after the first rejection are never tested.
