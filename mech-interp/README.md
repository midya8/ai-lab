# mech-interp

I trained a tiny transformer (2 layers, 2 heads, d=32) on an in-context copying task, then went looking for the induction head everyone talks about. Short version: it isn't there.

Training uses the autodiff engine in [`microgrid`](../microgrid). The analysis re-implements the forward pass in plain NumPy with hook points, and a test checks its logits match the trained model to 1e-9. On top of that: attention-pattern scores, head and MLP ablation, direct logit attribution, activation patching, composition scores and a causal copy test.

```bash
cd mech-interp
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                         # ~25 s
PYTHONPATH=src python -m mechinterp --seed 0 --steps 4000  # trains (~100 s) and prints the report
```

## The task

Each sequence repeats a block of 4-8 distinct tokens, with the period chosen per sequence. So "copy the token k back" doesn't work. The model has to find where the current token appeared before and copy whatever came after it, which is exactly the job an induction head is supposed to do.

All three seeds get 98-99% on the predictable positions.

## What I found

The copy test is the cleanest evidence. I overwrite the earlier copy of the answer with a token that appears nowhere else, and the model's prediction follows the edit 96-99% of the time (17-20% when I edit some other position instead). So it really is copying from the right place.

But it's not the textbook circuit:

- Layer-1 heads score below a uniform-attention baseline on the induction score. They mostly look at the previous token.
- K-composition from layer 0 to layer 1 is at random-matrix level.
- The lookup lives in layer 0. Ablating either layer-0 head drops accuracy from ~0.99 to 0.22-0.44. Ablating a layer-1 head costs much less.
- The MLPs do most of the direct writing to the correct logit. Knocking out the layer-0 MLP takes accuracy down to 0.19-0.44.
- Exactly how the work is split varies by seed.

So I can say where the work happens, but I haven't fully reverse-engineered the algorithm. Next I'd look at what the layer-0 MLP computes (patching or an SAE), and try an attention-only model to see if that forces a classic induction head.

Raw output is in `experiments/`. Three seeds, one synthetic task, one tiny model.
