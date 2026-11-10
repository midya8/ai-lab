# specdec

Speculative sampling (Leviathan et al. 2023; Chen et al. 2023) from scratch in numpy: a cheap draft model proposes k tokens, the target verifies them, and the output distribution is **exactly** the target's.

- `sampler.py`: `speculative_generate`, `autoregressive_generate`, `theoretical_alpha`, `expected_speedup`
- `models.py`: count-based n-gram LMs and a mixture LM for a draft of controllable quality
- Tests (6): exact length, identical models accept all, chi-square check that the joint over the first two tokens matches the target, fewer target calls with a good draft, alpha formula vs empirical, speedup bounds.

Run: `pip install -r requirements.txt && PYTHONPATH=src python -m pytest -q && PYTHONPATH=src python experiment.py`

## Experiment (synthetic 2nd-order Markov source, V=12, 3000 tokens)

```
w_target  k  alpha_emp  tok/target_call  modeled_speedup(c=0.1)
   0.00   2     0.253            1.51           1.10
   0.00   4     0.136            1.54           0.83
   0.00   8     0.067            1.53           0.60
   0.50   2     0.558            2.11           1.56
   0.50   4     0.412            2.65           1.20
   0.50   8     0.243            2.94           0.73
   0.80   2     0.791            2.58           2.01
   0.80   4     0.711            3.84           2.02
   0.80   8     0.578            5.62           1.31
   0.95   2     0.957            2.91           2.39
   0.95   4     0.925            4.69           3.08
   0.95   8     0.901            8.20           3.41
```

## Caveats
- Simulated: models are n-gram tables, not neural nets; "modeled speedup" assumes draft/target cost ratio c=0.1 and the iid-acceptance formula, not measured walltime.
- alpha_emp = accepted/proposed undercounts the true per-token acceptance, because proposals after the first rejection in a round are never tested. Treat it as a conservative proxy.
- The distribution test is statistical (chi-square, p=0.001 cutoff, fixed seeds).
- Finding: with a weak draft, larger k hurts (modeled speedup <1); gains need a draft close to the target.

## Adaptive k (v0.2)
`AdaptiveK(cost_ratio)` picks each round's k from an EMA of past acceptance, maximising `expected_speedup`. k depends only on earlier rounds, so output stays exactly the target distribution (chi-square test with the controller on). Use: `speculative_generate(..., k=0, controller=AdaptiveK(0.1))`. Run `PYTHONPATH=src python experiment_adaptive.py` (3 seeds, simulated cost = target_calls + 0.1*proposed):

```
w_target |  k=2   k=4   k=8  k=16 adaptive (mean k)
    0.00 | 0.809 0.904 1.163 1.650  0.815 (1.6)
    0.50 | 0.563 0.530 0.615 0.831  0.537 (3.8)
    0.80 | 0.459 0.363 0.322 0.378  0.332 (8.3)
    0.95 | 0.413 0.300 0.226 0.197  0.198 (15.3)
```
Adaptive stays within ~3% of the best fixed k without knowing the draft quality in advance (the cost ratio is assumed known). Still synthetic n-gram models; tests now 8.

## Multi-draft verification (v0.3)

`multi_draft_step(p, q, m, rng)` draws m i.i.d. candidates from the draft and tests them with recursive rejection sampling (residual `max(p-q,0)` after each rejection). The emitted token is exactly target-distributed (chi-square tested for m=1,2,4), and `multi_draft_acceptance` gives the closed-form chance that some candidate is accepted, which the tests check against simulation. Acceptance rises monotonically with m. This is a single-position, i.i.d.-candidate scheme (no token tree over multiple depths), and it reports acceptance only, not wall-clock.

## Draft trees (v0.4)
`tree_generate(target, draft, prompt, n, depth, m, rng)` verifies an m-ary draft tree level by level inside one target pass per round: m i.i.d. candidates per level are tested by recursive rejection, an accepted candidate's subtree is entered (up to `depth`), a full rejection ends the round with a residual sample, and a clean round adds a bonus token. Exactness is tested (chi-square on the joint of the first two tokens). `python experiment_tree.py` (3 seeds, depth 4, synthetic n-gram models), tokens per target call:

```
w_target | m=1 m=2 m=4 m=8
     0.2 | 2.01 2.99 4.30 4.84
     0.6 | 3.04 4.55 4.86 4.98
     0.9 | 4.58 4.94 5.00 5.00
```
Candidates help most for weak drafts and saturate at depth+1 = 5; the price is m draft nodes per level (about 6.4 per token at m=8), which this table does not charge for. Only the accepted path is expanded (not a full m^depth tree), and these are simulated models with no wall-clock.

## Cost accounting (v0.5)
`tree_cost_per_token(stats, n, cost_ratio)` = (target_calls + c * draft_nodes) / tokens, in units of one target pass (autoregressive = 1.0). `python experiment_cost.py` (3 seeds, depth 4) charges the draft nodes that the v0.4 table ignored. Result: **m=1 wins at every c in {0.02, 0.1, 0.3} and every draft quality tried** (e.g. c=0.1: 0.28 at m=1 vs 0.84 at m=8). Caveat, found while reading the output: the toy target is trained on i.i.d. uniform tokens, so it is near-uniform and even the weak drafts accept often; wide trees have little to buy here. This is a null result for wide trees on this toy, not evidence about real LLMs, where tree drafting is reported to help when draft and target diverge. A peaky target would be the right next test.
