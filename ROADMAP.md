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


## Next

1. rag-evals: (done hard benchmark) (done chunk sweep) -> next: dense embeddings beyond LSA
2. llm-eval-harness: (done) (done matrix) (done per-tag matrix) -> next: HTTP-model matrix specs
3. tiny-transformer follow-ups: (done v0.5, top-p, grad accumulation); (done clipping); (done dropout); (done weight tying); (done early stopping); (done repetition penalty); (done LR finder); (done label smoothing); (done per-position eval); (done sliding-window eval); (done lora-lab on its attention) (done RoPE) -> next: grouped-query attention or position interpolation for length extrapolation
4. agentflow follow-ups: (done shared memory) (done TTL) (done memory-aware Supervisor) (done failure fallback retries) (done circuit breaker) (done run metrics) (done Supervisor-in-DAG adapter) (done llm-router backend) -> next: async/streaming workers or a persistent run log
5. lora-lab: (done rank sweep) (done LoRA on tiny-transformer attention) -> next: multi-adapter hub (swap/blend adapters), QLoRA-style int8 base
6. logsentry: (done v0.6 Grafana dashboard)
7. ticket-triage: (done urgency head) (done active learning) (done multi-label tags) (done tag-aware routing) (done SLA-breach simulation) -> next: feed live router output into the simulator
8. llm-router: (done v0.1) -> (done latency tracking) (done semantic cache) (done learned complexity classifier) (done persistence)
9. promptlab: (done v0.1) -> v0.2 multi-sample done; (done GitHub Action + summary); next: LLM-judge check via evalkit
10. docqa: (done v0.1) -> (done paraphrase) (done broad benchmark) (done yes/no fixes) (done agentkit integration) (done agentflow worker) (done HTTP service) (done cache + metrics) next: docqa on llm-router / hybrid retrieval
11. autojob: agentic job-application assistant (only Midya's real resume data, never invents experience)
