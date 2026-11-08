# Roadmap

## Done

- rag-evals: BM25 retrieval + chunking + metrics + CLI (v0.1)

- tiny-transformer: NumPy GPT with manual backprop, Adam, sampling, tests (v0.1)

- llm-eval-harness: pluggable backends, 6 scorers, suites, baseline regression gate (v0.1)

- rag-evals: LSA dense retrieval + RRF hybrid + method comparison CLI (v0.2)

- agentflow: agent loop, DAG orchestrator, parallelism, retries (v0.1)

- llm-eval-harness: pass@k estimator + LLM-as-judge scorer (v0.2)

- logsentry: log anomaly detection (v0.1)

- ticket-triage: TF-IDF + softmax classifier, escalation (v0.1)

- tiny-transformer: KV-cache incremental decoding, equivalence-tested (v0.2)

- tiny-transformer: byte-level BPE tokenizer (v0.3)

- ticket-triage: k-fold CV, ECE, temperature scaling (v0.2)

- rag-evals: answer-faithfulness + answer-relevance eval (v0.3)

- agentflow: tracing spans + supervisor/router agent (v0.2)

- lora-lab: NumPy LoRA with manual backprop, merge/unmerge (v0.1)

- tiny-transformer: warmup+cosine LR, AdamW decay, BPE wired into train CLI (v0.4)

- logsentry: per-template spike detection (v0.2)

- logsentry: streaming detector + --follow CLI, batch-equivalence tested (v0.3)

- llm-eval-harness: GitHub composite Action + job summary (v0.3)

- logsentry: Prometheus exporter, --metrics-port (v0.4)

- logsentry: webhook alerts with cooldown (v0.5)

- agentkit: tool-calling agent loop, bounded steps, loop detection, dry-run/validate, JSON trace, pluggable provider (46 tests)

- ragkit: local RAG with chunking, TF-IDF/BM25 retrieval, cited answers, recall/citation-precision eval (59 tests)

- evalkit: golden sets, scorers, regression compare, Markdown/JSON reports, flaky detection (36 tests)

- guardkit: runtime guardrails — injection scoring, PII/secret detection with reversible redaction, policy file, redacted audit log, adversarial eval (62 tests; 24-case fixture: attack recall 1.000, PII caught 1.000, FPR 0.000). Runtime complement to promptguard (separate standalone repo), which lints prompt files statically.
- logsentry: Grafana dashboard generator + importable JSON (v0.6)
- tiny-transformer: self-contained checkpoints (config+tokenizer), held-out perplexity eval, generate/eval CLI (v0.5)
- llm-router: complexity scoring, cheapest/fastest/quality/balanced strategies, fallback chain, circuit breaker, LRU+TTL cache, rolling budget with degradation, verifier cascade, outage simulator CLI (39 tests)
- promptlab: prompt registry (hashed versions, tags, diff), strict templates, 9 check types, paired A/B compare (exact McNemar, Wilson, bootstrap CI), baseline regression gate with exit codes, offline RuleModel + cmd adapter (34 tests)
- promptlab v0.2: multi-sample runs (--samples/--threshold), flaky-case detection, backward-compatible run files (34 tests)
- tiny-transformer: top-p (nucleus) sampling shared module, CLI --top-p, cache/non-cache parity tested
- tiny-transformer: gradient accumulation (--micro-batch), full-batch-equivalence tested (33 tests)
- ticket-triage: urgency head (class-weighted logistic regression, recall-targeted threshold, combined queue routing; 17 tests; synthetic data)
- tiny-transformer: global-norm gradient clipping (--clip, logs pre-clip norm; 35 tests)
- tiny-transformer: dropout (--dropout; embeddings, attention, residuals; finite-difference-checked masks; 42 tests)
- tiny-transformer: weight tying (--tie-weights; shared wte/head, gradients summed, finite-difference-checked; 46 tests)
- tiny-transformer: early stopping (--patience/--eval-every/--min-delta, best-weights restore; 52 tests)
- tiny-transformer: repetition penalty (--repetition-penalty, CTRL-style; cached/non-cached parity tested; 56 tests)
- tiny-transformer: LR range test (--find-lr, restores weights, steepest-descent suggestion; 60 tests)
- tiny-transformer: label smoothing (--label-smoothing, train-only, eval stays plain NLL; finite-difference-checked; 63 tests)
- tiny-transformer: per-position loss eval (--by-position, context gain; 67 tests)
- tiny-transformer: sliding-window perplexity (--stride, each token counted once; exact vs manual full-context scoring; 72 tests)
- llm-router: observed-latency tracking (sliding-window median correction for fastest/balanced; 46 tests)
- llm-router: semantic cache (hashed lexical embeddings, cosine threshold, TTL/LRU; 53 tests)
- llm-router: semantic-cache persistence (atomic JSON save/load, age-based TTL across restarts; 65 tests)
- rag-evals v0.4: harder paraphrase benchmark (8 docs, 19 questions; recall@3 0.84, no longer saturated; 16 tests)
- promptlab: check --summary Markdown report + composite GitHub Action (31 tests)
- llm-eval-harness: multi-model comparison matrix (--matrix, winner + disagreement count; 14 tests)
- lora-lab: rank sweep (--sweep, ranks 1-16 vs full fine-tune; task saturates at 1.000, documented; 7 tests)
- agentflow v0.3: SharedMemory + attach_memory tools (search, atomic persistence, thread-safe; 15 tests)
- ticket-triage: active-learning review queue (--active; uncertainty sampling vs random; 10 new tests)
- rag-evals v0.5: chunk-size sweep (--sweep, Markdown table, best-by-MRR helper; 4 new tests)

- ticket-triage: multi-label tags (OvR logistic, per-tag tuned thresholds, micro/macro F1, Hamming, exact-match, --tags; 7 new tests)
- ticket-triage: tag-aware routing (queue + priority + SLA from category, urgency, tags; overridable rules; 7 new tests)
- ticket-triage: SLA-breach simulation (priority queues, staffing sweep, smallest staffing for a breach target; 6 new tests, synthetic)
- docqa v0.1: sentence-level cited answers on ragkit, coverage-based abstention, answer/abstain eval (9 tests)
- agentflow: SharedMemory TTL (default + per-entry ttl, lazy expiry, purge(), persisted expires_at; 2 new tests)
- docqa: paraphrase handling (synonym groups at 0.8 credit, --paraphrase toggle via constructor, paraphrase eval set; 3 new tests; generic words like 'back'/'within' deliberately excluded after they caused a wrong answer)
- llm-eval-harness: per-tag matrix (--matrix now adds a By-tag table with best backend per tag; 2 new tests)
- docqa: broad 13-question benchmark across all docs (answer 0.80, abstain 1.00; 2 yes/no misses documented; 1 new test)
- docqa: yes/no fixes (vague-quantifier filler, open/closed polarity group; broad benchmark answer 1.00, abstain 1.00; 2 new tests)
- agentflow: memory-aware Supervisor (recalls relevant memory into prompts, records results, reports memory_used; 2 new tests)
- agentflow: Supervisor fallback retries (max_attempts tries next workers, records failures in memory, failed_attempts in result; 2 new tests)
- agentflow: per-worker CircuitBreaker (open/half-open/closed, cooldown, Supervisor skips open workers, skipped_open in result; 3 new tests)
- agentflow: run metrics (per-task duration, critical path, parallelism, failure/retry counts; 2 new tests)
- agentflow: SupervisedAgent (Supervisor as a DAG node in Orchestrator; routing_summary; 4 new tests)
- docqa: agentkit integration (`ask_docs` tool via docqa.agent_tool.register; 5 new tests)
- docqa: agentflow worker (DocQAWorker, strict mode fails/skips downstream on abstention; 6 new tests)
- docqa: HTTP service (`python -m docqa serve`, POST /ask, GET /health, input validation; 4 new tests)
- docqa: answer cache + request metrics (LRU+TTL cache with X-Cache header, /stats JSON, /metrics Prometheus histogram, --cache-size/--cache-ttl; 16 new tests, 41 total)
- agentflow v0.4: llm-router backend (RouterBackend/RouterLLM, per-agent routing strategies, serialized thread-safe routing, per-agent/model cost report, spend cap, json_or_text verifier, real-router integration test; 10 new tests, 40 total)
- lora-lab: LoRA on tiny-transformer attention (GPTLoRA: adapters on qkv/proj[/fc/out] reusing the GPT's hand-written backprop via chain rule, merge/unmerge, adapter on/off, .npz adapter files; r=4 cuts domain-B NLL 5.69 -> 1.50 with 3.7% of weights vs full fine-tune 0.38 at 100% but heavy forgetting; 10 new tests, 17 total)
- tiny-transformer: rotary position embeddings (--rope; no wpe, inverse-rotation backward, pre-rotated KV cache, runs past block_size; finite-difference + parity tested; experiments/rope_vs_learned.py shows parity in-window and honest non-extrapolation at this scale; 89 tests)
- agentflow v0.5: persistent run log + resume (append-only JSONL journal, dependency-chained task fingerprints so edits invalidate only downstream work, crash-tolerant reads, compact, `python -m agentflow log`; demo_resume.py; 10 new tests, 50 total)
- lora-lab: adapter hub (swap/blend/route/compose; named adapters over one frozen GPT, delta blending, label-free routing by prompt NLL, SVD composition, save/load hub; second synthetic 'kitchen' domain; tech/kitchen specialists NLL 5.6 -> ~1.4, routing 1.000, 50/50 blend is a compromise; 13 new tests, 30 total)
- docqa: corpus-grounded spelling correction (OSA distance 1-2 against document vocabulary only, never invents words, deterministic ties, reported corrections, --no-spell; typo benchmark answered 7/12 -> 12/12 with abstention 8/8 unchanged and clean benchmarks identical; 23 new tests, 64 total)
- lora-lab: QLoRA-style quantized base (block-wise absmax int8 / uniform int4 / NF4 codebook, nibble-packed save files, `QuantizedBase` keeps codes and verifies the base stays frozen; GPTLoRA trains through dequantized weights unchanged; 4.3x smaller model, int8 weight error 0.6%, NF4 degrades base less than int4 (0.373 vs 0.396 NLL), adapter quality on domain B matches fp32 within noise; 20 new tests, 50 total)
- promptlab v0.3: LLM-as-judge checks and judge calibration (strict 1-5 reply parsing, injection-defanged grading prompt, `votes` lower-median, judge failures fail loudly; `calibrate` reports agreement + Wilson CI, Cohen's kappa, Spearman, bias, lenient/harsh confusion, `--min-kappa` gate; Action `judge` input; offline lexical RuleJudge scores only kappa 0.44 on 32 author-written labels, reported as is; 33 new tests, 64 total)
- promptlab v0.4: pairwise LLM-judge A/B with a position-swap bias check (every pair judged in both orders and reconciled: agree / position flip / hesitant; win rate + Wilson CI + exact sign test; flips tested for first- vs second-slot lean; `pairwise` and `pair-audit` commands with exit-code gates; audit vs 24 author-written human preferences: lexical stand-in 79.2% reconciled accuracy with zero bias by construction, simulated biased judge caught at 29% consistency; 30 new tests, 94 total)
- tiny-transformer: grouped-query / multi-query attention (--kv-heads; compact KV cache shrinks n_head/n_kv_head-fold, expand/reduce adjoint backward, finite-difference-checked, MQA == tied-head MHA exactly, cached-decode parity; experiments/gqa_vs_mha.py over 3 seeds shows mixed quality, reported as noise; 107 tests)
- tiny-transformer: context extension for RoPE models (`extend_context` views: position interpolation, NTK-aware scaling, plain-extrapolation baseline; shared weights, scale measured against the trained length, `with_window`, `finetune_extended`, `extend` CLI, checkpoint round trip; gradient-checked, KV-cache parity incl. GQA; experiments/context_extension.py: zero-shot extrapolation costs 1.4/2.1 nats at 2x/3x, NTK helps at 2x only, unfine-tuned linear PI is worst; fine-tuned columns reported as memorisation-limited; 131 tests)
- lora-lab: LoftQ-style init (`loftq_init`: alternating quantize / rank-r SVD so `Q + s·A·B` approximates fp32 W; `QuantizedBase(source=)` hook; weight error ~20% below plain nf4/int4 at rank 4 and falling with rank/iterations, but downstream NLL gains are small and mostly noise at this scale, reported as such; `loftq_demo` CLI; 16 new tests, 66 total)
- visionlab v0.1 (new project): NumPy CNN with im2col Conv2D and hand-written backprop (gradient-checked), Adam/SGD, procedural 6-class shapes dataset, saliency/SmoothGrad/Grad-CAM/occlusion with a deletion-curve faithfulness metric (occlusion 0.28 < SmoothGrad 0.35 < saliency 0.37 < Grad-CAM 0.59 < random 0.69), FGSM/PGD attacks, natural-perturbation report, FGSM adversarial training (FGSM@0.05 acc 0.33 -> 0.72, 3 seeds; no transfer to noise/shift/contrast), PNG montage writer; 41 tests, synthetic data
- visionlab v0.2: BatchNorm2D (batch/running stats, saved with the model) and GlobalAvgPool, `--head gap --batchnorm`; gradient-checked in both modes; heads_shift.py (3 seeds): dense head 0.98 -> 0.37 accuracy from 0 to 4 px shift, width-16 GAP+BN 0.98 -> 0.93, but contrast-0.5 accuracy 0.66 vs 0.90, reported as a trade-off; BN helps GAP, not dense (49 tests)
- visionlab v0.3: certified robustness by interval bound propagation (`certify.py`: sound logit bounds through conv/BN/ReLU/max-pool/GAP/dense, margin-based certificates, certified radius by bisection, IBP backward pass + `train --ibp-eps`, `certify` CLI; 3 seeds: standard and FGSM-trained nets certify 0% at eps 0.01-0.05, IBP-trained certifies 56% at eps 0.03 / 31% at 0.05 for ~15 points of clean accuracy, PGD upper bounds alongside; finite-difference-checked, bounds enclose PGD/corner points; 15 new tests, 64 total)


## Next

1. rag-evals: (done hard benchmark) (done chunk sweep) -> next: dense embeddings beyond LSA
2. llm-eval-harness: (done) (done matrix) (done per-tag matrix) -> next: HTTP-model matrix specs
3. tiny-transformer follow-ups: (done v0.5, top-p, grad accumulation); (done clipping); (done dropout); (done weight tying); (done early stopping); (done repetition penalty); (done LR finder); (done label smoothing); (done per-position eval); (done sliding-window eval); (done lora-lab on its attention) (done RoPE) (done GQA/MQA) (done context extension: PI + NTK) -> next: a key-value recall benchmark with a larger model to test long-context use properly
4. agentflow follow-ups: (done shared memory) (done TTL) (done memory-aware Supervisor) (done failure fallback retries) (done circuit breaker) (done run metrics) (done Supervisor-in-DAG adapter) (done llm-router backend) (done persistent run log + resume) -> next: async/streaming workers
5. lora-lab: (done rank sweep) (done LoRA on tiny-transformer attention) (done adapter hub) (done QLoRA-style quantized base) -> (done LoftQ-style init) -> next: double quantization (quantize the per-block scales), or a QLoRA memory-vs-quality frontier plot
6. logsentry: (done v0.6 Grafana dashboard)
7. ticket-triage: (done urgency head) (done active learning) (done multi-label tags) (done tag-aware routing) (done SLA-breach simulation) -> next: feed live router output into the simulator
8. llm-router: (done v0.1) -> (done latency tracking) (done semantic cache) (done learned complexity classifier) (done persistence)
9. promptlab: (done v0.1) -> v0.2 multi-sample done; (done GitHub Action + summary); (done LLM-judge checks + calibration) (done pairwise judge + position-swap bias audit) -> next: pairwise results fed into the regression gate, or multi-judge ensembles with agreement checks
10. docqa: (done v0.1) -> (done paraphrase) (done broad benchmark) (done yes/no fixes) (done agentkit integration) (done agentflow worker) (done HTTP service) (done cache + metrics) (done typo-tolerant spelling) next: docqa on llm-router / dense or hybrid retrieval
11. autojob: agentic job-application assistant (only Midya's real resume data, never invents experience)
12. visionlab follow-ups: (done GAP head vs dense under shift) (done batch-norm) (done certified robustness via IBP) -> next: contrast/photometric augmentation to fix the GAP head's contrast brittleness, tighter bounds (CROWN / zonotope) to certify the FGSM-trained net, IBP-compatible BatchNorm
