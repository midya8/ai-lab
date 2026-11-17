# ml-from-scratch

Where I build ML ideas myself to understand them. Mostly NumPy or plain Python, small enough to run on a laptop CPU, and each folder has its own README, tests and a few experiments. A second group is practical LLM tooling: evals, RAG, agents, routing, guardrails.

Every project runs on its own. The usual pattern is:

```bash
cd <project>
pip install -r requirements.txt     # if there is one (usually just numpy + pytest)
PYTHONPATH=src python -m pytest -q
```

## Models and training

| project | what it is |
|---|---|
| [tiny-transformer](tiny-transformer) | GPT in NumPy with hand-written backprop, RoPE, GQA, KV cache |
| [microgrid](microgrid) | small autodiff engine that grew into a trainable MiniGPT, plus weight quantization and GPTQ |
| [lora](lora) | LoRA, QLoRA-style quantized bases, LoftQ init and an adapter hub on the tiny GPT |
| [flash-attention](flash-attention) | FlashAttention, split-KV decoding and a paged KV cache, exact vs naive |
| [mixture-of-experts](mixture-of-experts) | MoE routing, load balancing and when sparsity beats dense |
| [mamba-numpy](mamba-numpy) | selective state-space model with backprop through the scan |
| [specdec](specdec) | speculative decoding experiments, n-gram and tiny transformer pairs |
| [bpe-tokenizer](bpe-tokenizer) | byte-level BPE vs a unigram LM tokenizer |
| [bpekit](bpekit) | the practical BPE tokenizer: offsets, chunking, packing, export |
| [diffusion-2d](diffusion-2d) | DDPM/DDIM on 2-D data where the optimal model is known |
| [normalizing-flows](normalizing-flows) | RealNVP/NICE flows scored against exact densities |
| [gnn](gnn) | GCN, GraphSAGE, GAT, APPNP and when graphs help |
| [cnn-explain](cnn-explain) | small CNN, saliency maps, attacks and certified robustness |
| [voicekit](voicekit) | offline voice-command pipeline on synthetic speech |
| [rlgrid](rlgrid) | tabular RL and a NumPy DQN on a gridworld |
| [mcts](mcts) | MCTS for tic-tac-toe/Connect-N, plus a tiny AlphaZero |

## Understanding models

| project | what it is |
|---|---|
| [mech-interp](mech-interp) | took apart a tiny copying transformer and didn't find an induction head |
| [sparse-autoencoders](sparse-autoencoders) | five SAE variants scored on features I know the truth for |
| [hopfield](hopfield) | classical vs modern Hopfield nets, and attention as retrieval |
| [ntk](ntk) | neural tangent kernel, and when wide nets act like kernel regression |
| [gaussian-processes](gaussian-processes) | GPs, sparse approximations and where they mislead |
| [bayes-opt](bayes-opt) | Bayesian optimisation on the standard test functions |
| [conformal](conformal) | conformal prediction, and when 90% coverage is misleading |
| [datadoctor](datadoctor) | finding label errors, duplicates, leakage and drift in tabular data |

## LLM tooling

| project | what it is |
|---|---|
| [evalkit](evalkit) | golden-set regression checks for prompts and models |
| [llm-eval-harness](llm-eval-harness) | eval harness with pluggable backends and a CI action |
| [prompt-regression](prompt-regression) | versioned prompts, paired A/B stats, judge calibration |
| [rag-evals](rag-evals) | BM25, LSA and hybrid retrieval with recall/MRR/nDCG |
| [ragkit](ragkit) | local RAG with line-level citations, standard library only |
| [docqa](docqa) | extractive Q&A that says "I don't know" |
| [vecdb](vecdb) | IVF, PQ and HNSW vector search with filters |
| [codesearch](codesearch) | search a Python repo for functions in plain English |
| [agentkit](agentkit) | minimal agent runner with real guardrails |
| [agentflow](agentflow) | multi-agent DAGs with retries, resume and tool journaling |
| [llm-router](llm-router) | cost-aware routing, fallbacks, caching and budgets |
| [guardkit](guardkit) | runtime prompt-injection and PII guardrails |
| [reviewkit](reviewkit) | PR review bot that checks the LLM's comments against the diff |
| [ticket-triage](ticket-triage) | support-ticket classifier and routing |
| [logsentry](logsentry) | log anomaly detection with streaming and Prometheus |
| [autojob](autojob) | job-search helper that won't make things up about you |

A few of these overlap on purpose. `rag-evals` only does retrieval and `ragkit` adds answers on top. `evalkit` and `llm-eval-harness` are two takes on the same idea.

Nothing needs an API key. Where a real model is optional, it's opt-in through an environment variable. [ROADMAP.md](ROADMAP.md) has what I want to try next.
