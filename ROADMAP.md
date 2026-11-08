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

## Next

1. rag-evals: larger/harder benchmark (current corpus saturated at 1.0)
2. llm-eval-harness: (done) -> next: multi-model comparison matrix
3. tiny-transformer follow-ups: (done v0.5, top-p, grad accumulation); (done clipping); (done dropout); (done weight tying); (done early stopping); (done repetition penalty); (done LR finder); (done label smoothing); (done per-position eval); (done sliding-window eval); next: lora-lab on tiny-transformer attention
4. agentflow follow-ups: shared memory
5. lora-lab: LoRA on tiny-transformer attention, rank sweep
6. logsentry: (done v0.6 Grafana dashboard)
7. ticket-triage: (done urgency head) -> next: multi-label tags, active-learning review queue
8. llm-router: (done v0.1) -> (done latency tracking) (done semantic cache) (done learned complexity classifier) next: semantic-cache persistence
9. promptlab: (done v0.1) -> v0.2 multi-sample done; next: LLM-judge check via evalkit, GitHub Action for check
10. docqa: document Q&A CLI built on ragkit + agentkit, citations, no-hallucination mode
11. autojob: agentic job-application assistant (only Midya's real resume data, never invents experience)
