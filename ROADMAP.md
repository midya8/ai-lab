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

## Next
1. rag-evals: answer-faithfulness eval, larger/harder benchmark (current corpus saturated at 1.0)
2. llm-eval-harness: GitHub Action wrapper
3. tiny-transformer follow-ups: BPE tokenizer, LR schedule
4. agentflow follow-ups: shared memory, supervisor/router agent, tracing
5. LoRA fine-tuning pipeline
6. logsentry follow-ups: per-template rates, streaming, Prometheus exporter
7. ticket-triage: urgency head, k-fold CV, calibration
