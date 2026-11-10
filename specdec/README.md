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

## Peaky target (v0.6)
`python experiment_peaky.py`: target is a random 16-token Markov chain with Dirichlet(0.15) rows (mean entropy 1.59 nats vs 2.77 uniform); drafts are tempered copies of the target or mixes with another chain. Cost/token at c=0.1, depth 4, 3 seeds, m in (1,2,4,8):

| draft | m=1 | m=2 | m=4 | m=8 |
|---|---|---|---|---|
| temper T=1.5 | **0.356** | 0.401 | 0.535 | 0.846 |
| temper T=3.0 | **0.482** | 0.509 | 0.576 | 0.895 |
| temper T=6.0 | 0.579 | **0.545** | 0.634 | 0.917 |
| mix w=0.9 | **0.318** | 0.372 | 0.522 | 0.840 |
| mix w=0.6 | 0.454 | **0.427** | 0.545 | 0.851 |
| mix w=0.3 | 0.600 | **0.594** | 0.678 | 0.914 |

Finding: on a peaky target, m=2 beats m=1 once the draft is poor (acceptance roughly below the point where a second candidate pays for its extra node), by up to ~6%; m>=4 never wins, because cost grows linearly with nodes. A modest, honest gain, only with a bad draft, small sample (3 seeds, no error bars). Still toy models and an idealised cost model.


## Real transformer pair (v0.7)
`specdec.neural.MiniGPTLM` wraps a trained microgrid `MiniGPT` (last `ctx` tokens -> next-token probs) so it can be a target or draft; exactness is tested with a neural pair (chi-square on the first token). `PYTHONPATH=src python experiment_neural.py 300` trains a target (d=48, 2 layers, 67k params) and a draft (d=24, 1 layer, 12.6k params) on stdlib source chars (300 steps, val nll 2.90 / 3.05) and measures (results file: `experiment_neural_results.txt`, 3 seeds, 120 tokens):

```
 k  alpha_emp  tok/target_call  cost/token
 1     0.714        1.70           0.875
 2     0.678        2.33           0.846
 4     0.429        2.68           1.103
 8     0.301        3.28           1.456
```
Finding: the draft has 0.19x the parameters but its measured forward time is **0.49x** the target's (numpy overhead dominates tiny nets), so with c=0.49 the best case is ~15% cheaper per token (k=2) and k>=4 is worse than plain sampling. Gains need a draft that is much cheaper than the target, not just smaller. Caveats: undertrained models, no KV cache, the adapter calls the target once per verified position (so measured wall-clock is slower than autoregressive; the cost column uses the one-batched-pass-per-round model), 3 seeds, no error bars.


## Prompt-lookup draft (v0.8)
`specdec.PromptLookupLM(vocab, n, conf)` is a model-free draft: if the last n tokens appeared earlier in the context, it puts `conf` mass on the token that followed the latest match, else it is uniform. It depends only on the context, so output stays exactly target-distributed (chi-square test in `tests/test_lookup.py`). `python experiment_lookup.py` (k=4, 200 tokens, 3 seeds, order-2 n-gram targets):

| target | draft | alpha | tokens/target call |
|---|---|---|---|
| periodic (i mod 8) | lookup | 0.800 | 4.16 |
| periodic | unigram | 0.037 | 1.15 |
| random | lookup | 0.158 | 1.62 |
| random | unigram | 0.856 | 4.38 |

Finding: lookup wins big on copy-heavy text and loses on text with no repeats (the unigram draft matches a uniform target by construction). Toy targets; alpha on the periodic target is capped by conf=0.9.

## Backoff prompt-lookup draft

`BackoffLookupLM(vocab, n_max, n_min)` tries the longest matching context suffix first and falls back to shorter ones, with confidence rising with match length. Exactness is preserved (tested by chi-square). `experiment_backoff.py` (synthetic repeated-phrase target, 20 seeds, k=4): acceptance n=1 0.071, n=2 0.292, n=3 0.243, backoff 3..1 0.287. Result: backoff matches the best fixed n within noise rather than beating it; its value is not having to tune n (n=1 and n=3 are both clearly worse). Confidence constants were not tuned.
