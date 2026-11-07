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

## Next
1. rag-evals: larger/harder benchmark (current corpus saturated at 1.0)
2. llm-eval-harness: (done) -> next: HTML report
3. tiny-transformer follow-ups: checkpoint tokenizer with model, eval perplexity
4. agentflow follow-ups: shared memory
5. lora-lab: LoRA on tiny-transformer attention, rank sweep
6. logsentry follow-ups: Grafana dashboard JSON, alert webhooks
7. ticket-triage: urgency head
