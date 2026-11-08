# ai-lab

Midya's AI lab: small, tested, runnable AI projects.

| Project | Pitch |
|---|---|
| [SpectralLab](https://github.com/midya8/spectrallab) | Manual normalized graph clustering, complete degree-response derivatives and basis-invariant projector sensitivity. 33 Python/nine Node tests; 1,728 original point states plus 168 sensitivity states. [Graph inspector](https://midya8.github.io/spectrallab/), [finite-refit sensitivity](https://midya8.github.io/spectrallab/sensitivity.html). Tied rounding and refused boundaries stay visible. Original synthetic full-data illustration, not prediction, a label gradient or exact discrete optimization. |
| [ForestLab](https://github.com/midya8/forestlab) | Binary CART forests with exact Gini ordering and complete bootstrap/OOB membership witnesses. 20 Python/five Node tests; 992 actual browser paths. Stronger 5NN stays visible: 0/270 test errors versus selected forest 3/270. [Live inspector](https://midya8.github.io/forestlab/). Historical grouped descriptors, not authentication or calibrated confidence. |
| [RatingLab](https://github.com/midya8/ratinglab) | Observed-only ALS, strong winning bias control and finite conditional ridge edit/deletion responses with saved-vs-re-solved baselines, leverage and independent direct refits. 30 Python/nine Node tests. [Conditional response inspector](https://midya8.github.io/ratinglab/influence.html). Frozen-factor response, not full retraining, unlearning or causal influence. |
| [MarginLab](https://github.com/midya8/marginlab) | Manual linear/RBF dual SVM and exact represented-linear weak-duality proofs after equality repair; independent Fraction/BigInt readers and separate source matching. 29 Python/nine Node tests. [Live proof inspector](https://midya8.github.io/marginlab/certificate.html). Historical educational data, not clinical validation or authentication. |
| [rag-evals](rag-evals) | From-scratch BM25, LSA and RRF-hybrid RAG retrieval with chunking and Recall/MRR/nDCG evaluation harness |
| [agentkit](agentkit) | Dependency-free LLM agent runner: tool-calling loop, bounded steps, loop detection, dry-run, JSON trace — 46 tests |
| [ragkit](ragkit) | Local RAG over your own docs: chunking, TF-IDF/BM25 retrieval, answers with citations, eval — 59 tests |
| [evalkit](evalkit) | LLM eval harness: golden sets, scorers, regression compare, Markdown/JSON reports — 36 tests |
| [guardkit](guardkit) | Runtime guardrails for LLM pipelines: injection scoring, PII/secret redaction, policy file, audit log, adversarial eval — 62 tests |
| [tiny-transformer](tiny-transformer) | GPT from scratch in pure NumPy with hand-written backprop (gradient-checked), trainer and sampler |
| [llm-eval-harness](llm-eval-harness) | Pluggable LLM eval harness with scorers, prompt suites and a CI regression gate |
| [agentflow](agentflow) | Multi-agent orchestration framework: tool-using agents, task DAG, parallel runs, retries |
| [logsentry](logsentry) | Log anomaly detection: template mining, robust volume spikes, error bursts, new-pattern alerts, per-template spikes, streaming mode, Prometheus exporter, webhook alerts |
| [ticket-triage](ticket-triage) | Support-ticket triage classifier from scratch: TF-IDF + softmax regression with confidence-based escalation |
| [lora-lab](lora-lab) | LoRA low-rank adaptation from scratch in NumPy: manual backprop, gradcheck, merge/unmerge |
| [llm-router](llm-router) | Cost/latency-aware LLM router: complexity scoring, 4 strategies, fallback chains, circuit breakers, exact + semantic cache, budget guardrails, cascade, outage simulator — 53 tests |
| [promptlab](promptlab) | Prompt versioning + paired A/B testing (exact McNemar, Wilson/bootstrap CIs) + baseline regression gate + GitHub Action for CI — 31 tests |
| [docqa](docqa) | Grounded document Q&A: verbatim cited sentences, coverage-based abstention ("I don't know"), synonym-aware paraphrase matching, HTTP service with answer cache + Prometheus metrics — 41 tests |

Notes on overlap: `rag-evals` is retrieval-only evaluation; `ragkit` adds answering with citations on top of retrieval. `evalkit` is a general prompt/model eval harness; `rag-evals` is RAG-specific, and `llm-eval-harness` is the pluggable-backend variant. `guardkit` is runtime input/output guardrails; `promptguard` (separate repo) is the static file linter for prompt files.

Each project is self-contained with its own README, tests, and CI. All run offline with no API keys — a stub provider is used for demos and tests; a real LLM is opt-in via an environment variable.

See [ROADMAP.md](ROADMAP.md) for what's next.
