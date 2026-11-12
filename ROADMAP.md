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

- reviewkit v0.1 (new project): PR review bot — unified-diff parser with new-file line tracking, 21 rules (AST + tokenizer code view so strings/comments/docstrings don't trigger; secrets never echoed; multi-line `torch.load`/`yaml.load` handled), `.reviewkit.json` + inline `reviewkit: ignore[...]`, optional LLM pass (nonce-fenced untrusted diff, strict JSON, lines validated against added lines, severity capped at high, @mention/HTML-marker defusing, rule/model dedupe, OpenAI-compatible provider tested against a local HTTP server), fingerprint-based idempotent posting with 422 body-only fallback, composite GitHub Action (not run on GitHub), `bench` harness: dev set 38/0/0, held-out first run P 0.76 R 0.67 (3 of the misses were my label mistakes), real gaps fixed afterwards so that set is no longer held-out; dogfooding found a `def eval` false positive; 73 tests

- codesearch v0.1 (new project): semantic code search for Python repos — AST chunking (function/method/class/module, comments and decorators included), identifier-aware terms (camel/snake/acronym + small stemmer), BM25F over six weighted fields, NumPy randomized-SVD LSA and RRF hybrid, exact-symbol pinning for identifier queries, test-code demotion, SHA-256 incremental on-disk index (only changed files re-parsed), match explanations, `similar` (query by example) and `dups` (LSA cosine AND term-set Jaccard); two 26-query labelled sets (dev / held-out, author-written after reading the function list): BM25F MRR 0.70 dev / 0.59 held-out vs name-only 0.25 / 0.12, flat BM25 lead mostly vanishes on held-out, LSA and hybrid do NOT beat BM25F (reported); dogfooding `dups` finds real copy-paste in tiny-transformer experiments; 62 tests
- lora-lab: double quantization (`double_quant=True`: block scales mean-centred and stored as int8 with fp32 second-level scales per 256, weight codes chosen against the dequantized scales, int8 scale codes in `.save()` files with backward-compatible loading; 4.5 → 4.13 bits/weight at block 64 with unchanged weight error, matching the QLoRA paper; `dq_frontier` over blocks 16-256 x nf4/int4 x 3 seeds: nf4 block 16 + DQ has 13% lower weight error than plain block 64 at the same ~4.5 bits, NLL differences are noise and reported as such; 20 new tests, 86 total)

- datadoctor v0.1 (NEW): confident-learning label-error detection on cross-validated probabilities, duplicates / conflicting duplicates / leakage / MinHash text near-dupes / per-class outliers, audit CLI + CI gate, benchmark on injected noise (41 tests)

- autojob v0.1: agentic job-application assistant — profile validation (strict, every error at once), posting parse, transparent match scoring (required 0.6 / preferred 0.25 / experience 0.15, never 1.0-by-default), facts-only cover-letter drafts (only matched skills named), review notes listing gaps to never claim, JSONL application tracker with company+title dedupe and applied→screening→interview→offer→hired state machine; never submits, drafts only; 49 tests, example profile/posting fictional
- autojob v0.2: status history on every tracker entry and funnel analytics in `report` — stage conversion rates, response/rejection rates, median days in stage (completed stays only), match-score averages split by outcome, stale applications (--stale-days, default 14), per-company roll-up, --json/--today flags (66 tests)
- autojob v0.3: interview-prep packets — `autojob prep` builds a study sheet from profile/posting facts only (talking points with years/level + verbatim evidencing bullets, unevidenced skills flagged, gaps, likely questions, negotiation facts, verbatim story bank most recent first); `apply` also writes interview-prep.md into every packet (77 tests)
- autojob v0.4: follow-up reminders — `autojob followup` lists applications quiet past their stage's wait (7 days applied/screening, 3 interview, 2 offer; --wait-days overrides all; exactly N days is not due yet) with a facts-only draft email (role, company, date applied + your name; "Hiring team" greeting, no recruiter name invented); `followup --record` logs a sent follow-up as a history event, resetting the follow-up and stale clocks, refusing same-day duplicates and terminal applications; follow-ups recorded mid-stage do not split stage stays in analytics (93 tests)
- autojob v0.5: tailored resume sections — `autojob tailor` builds the facts a resume needs for one posting: a skills matrix covering every skill the posting asks for (have it: years/level + verbatim evidencing bullets; gap: flagged "do not claim", never listed as a claim), skills to lead with (matched skills sorted by years, most first, required before preferred), experience roles ranked by how many distinct matched posting skills their bullets mention (most recent first on ties; unevidenced roles rank last and are flagged), summary ingredients (headline verbatim + top lead skills, not a written summary), education verbatim; `apply` also writes tailored-resume.md into every packet (107 tests)
- autojob v0.6: weekly digest + gap leaderboard — `autojob digest` is one page over the whole log (funnel, median days in stage, match-score summary, stale applications, due follow-ups, and the skills postings keep asking for that the profile lacks, most-missed first, required/preferred counted separately, companies listed); `apply` now records the posting's required/preferred skills and which the profile matched in the tracker entry, and entries logged before v0.6 are counted as skipped, not guessed (123 tests)
- autojob v0.7: gap-closing plan — `autojob plan` turns the digest's gap leaderboard into work: the skills the profile has but no bullet proves (name, years/level, which postings asked, most recent role suggested as the place to add a real bullet; the plan never writes one), the skills to learn (leaderboard reused verbatim, "do not claim until you have it"), and `quick_wins`, the three fixes that help the most logged applications (143 tests)
- autojob v0.8: posting ranking — `autojob rank` scores every `.md`/`.txt` posting in a folder against the profile with the same transparent match as `match`, best first (score, then required coverage, then title); postings already in the application log are left out by default and flagged with their application id under `--include-applied`; the "best next application" line always names the highest-ranked posting not yet applied; a posting with no recognisable skills scores `None` and sorts last, labelled unmeasurable; an unreadable file is reported in `skipped`, never fatal (162 tests)
- autojob v0.9: salary analysis — `autojob salary` compares what a folder of postings pays against the salary target in the profile's preferences, best-paying first: per-posting verdicts (meets / below / above target, no range listed, not compared with the reason), per-currency medians over ranged non-hourly postings, already-applied postings flagged with their application id; currencies are never converted (a range is compared only when it states the same three-letter code as the target) and hourly rates are listed but never compared with an annual target; `parse_salary` now captures a currency code written after the figures (189 tests)
- autojob v0.10: offer comparison — `offer --record --id APP-001 --base 130000 --currency CAD` stores an offer on a tracker entry (base, annual bonus, equity with an optional vesting schedule, paid time off, work mode, start date, decision deadline, note); a base stated hourly is annualised at 2,080 hours a year, monthly at 12 months, and the stated figure is kept next to the annualised one; recording is refused on a terminal application and does not move the status. `offer` on its own compares every recorded offer against the salary target in the profile's preferences: one row per application using its latest offer, ranked by first-year total (base + bonus + equity per vesting year) among offers in the target's currency, with a verdict on the annual base; currencies are never converted, equity counts only with a vesting schedule, and each row shows days to the decision deadline and flags one already passed (219 tests)
- autojob v0.11: interview rounds log — `interview --record --id APP-001 --date 2026-11-12 --round "Hiring manager" --interviewer "Priya Nair" --mode video --duration-minutes 45` stores a round on a tracker entry (date, round label, interviewer, mode, duration, note); validation reports every problem at once; recording is refused on a terminal application and does not move the status, and several rounds per application are allowed, including two on the same day. `interview` on its own lists the week ahead: upcoming rounds soonest first (today counts as upcoming) and past rounds most recent first, each with its days to go or days since; `--id` shows one application's rounds only; everything recorded is a fact the user supplies, never generated text (254 tests)
- autojob v0.12: calendar export — `autojob calendar --out search.ics` exports the next 14 days as an .ics file: interview rounds in the window, offer decision deadlines in the window, and follow-ups due as of today (dated today, because the action is due now). A round with a start time exports as a timed event (45 minutes from 14:30 ends 15:15); a round with a time but no recorded duration is shown as one hour and the description says so; a round with no time is an all-day event. Times are wall-clock times exactly as recorded, no timezone stored and none invented, so the file uses floating local times. Deadlines and interviews on terminal applications are left out. UIDs are a hash of kind + application + date + summary, so re-exporting the same log produces the same file. `interview --record` gained an optional `--time HH:MM` (validated 24-hour, normalised, e.g. 9:30 → 09:30). RFC 5545 text escaping and 75-octet line folding that never splits a multi-byte character (283 tests)
- autojob v0.13: contacts log — `contacts --record --id APP-001 --name "Sam Chen" --role Recruiter --channel linkedin --date 2026-11-01` stores a person on a tracker entry (name, role, channel, date first reached or last spoke, note); the date is required and never defaulted, and there is deliberately no email or phone field. Recording the same person again is a new touch, not a duplicate: the list shows one row per person with the latest touch, touch count, and first-contact date. `contacts` on its own lists everyone on a live application, quietest first — quiet means not spoken to *that person* for longer than `--quiet-days` (default 14; exactly N days is not quiet yet), and an interview with someone else or an offer arriving does not count. Recording a contact is refused on a terminal application, does not move the status, counts as activity for follow-ups and the stale check, and never splits a stage stay in the analytics (316 tests)
- autojob v0.14: activity timeline — `timeline` lays out everything the other commands already recorded as one feed across all applications, newest first: applications (with the match score), status moves, follow-ups, interviews, contacts, and offers, grouped by day. An interview line names the round, interviewer, mode, duration, and start time exactly as recorded; an offer line states the base, bonus, and equity exactly as recorded. Details are paired to their records, not assumed — an interview event by date and round label, a contact event by date and name, the n-th offer event to the n-th offer in the list — and when a record is missing the line falls back to the bare fact instead of guessing figures; an event with a bad date or an unknown kind is skipped, never fatal. `--since` / `--until` bound the feed by date (inclusive), `--kind` filters to one kind of event and repeats, `--id` shows one application, `--oldest` flips the feed to chronological (340 tests)
- autojob v0.15: posting closing dates — `deadlines` lists the closing date of every live application, soonest first (overdue / today / closing soon 1–7 days / later), recorded at apply time (`apply --deadline YYYY-MM-DD`) or later (`deadlines --record --id APP-001 --date YYYY-MM-DD`); re-recording moves it. A past date is allowed and flagged overdue, not refused. `--days N` narrows to N days out or sooner (overdue always shown), `--overdue` keeps only overdue rows, and the header counts live applications with no closing date recorded instead of inventing one. A closing date is a field on the tracker entry, not a history event: recording one does not reset the follow-up or stale clocks and does not appear in the timeline feed. Posting closing dates in the window also export as all-day events from `calendar` (374 tests)
- ticket-triage: label audit via datadoctor confident learning (3 new tests, 51 total)
- datadoctor: self-contained HTML audit report (`--html`), 3 tests
- datadoctor: instance-dependent (border-concentrated) noise injection + experiment; recall drops 0.86 -> 0.23 and noise rate is underestimated 0.20 -> 0.06 (4 tests, 54 total)
- tiny-transformer: associative key-value recall task + benchmark (RoPE reaches 0.98 on 2 pairs, learned positions 0.79; 4 pairs unconverged)
- datadoctor: regression-target issue detection via out-of-fold residuals with local robust MAD scaling (6 tests, 64 total)
- datadoctor: ordinal-target issue detection (`find_ordinal_issues`, gap measured in levels, suggested level) + `inject_ordinal_noise` (2 tests, 66 total)
- datadoctor: multi-label issue detection (per-label confident learning, missing vs spurious tags, ranked cells; synthetic precision/recall ~0.85-0.91, easy linear case disclosed; 7 tests, 73 total)
- agentflow v0.6: `AsyncOrchestrator` (asyncio DAG runner; coroutine or sync LLMs/tools, concurrency semaphore, per-attempt timeout, `stream()` in completion order, cancellation; 11 tests, 61 total)
- agentflow v0.7: `RunLog` resume for `AsyncOrchestrator` (`run`/`stream`/`run_sync` take `log=`, `resume=`; fingerprint-matched replay, journal shared with the sync runner, finished tasks journaled before yielding so abandoning a stream loses nothing; 5 new tests, 66 total)
- promptlab: pairwise head-to-head gate (`pairwise --gate`, `--max-loss-rate`): fails when the candidate is significantly worse than the baseline or loses too many pairs; the loss-rate rule covers small sets where significance is unreachable (6 tests, 112 total)
- promptlab: gate on ensemble verdicts (`ensemble_gate`; `--gate`/`--max-loss-rate` apply to the majority outcome, `--min-kappa-ensemble` rejects low-agreement verdicts; 3 tests, 115 total)
- promptlab v0.6: cost metering (`--price IN,OUT`) and hard budgets (`--budget`, exit 3, aborts before the call, never counted as a model failure); estimated tokens disclosed (6 tests, 121 total)
- promptlab v0.7: HTTP-backed models and judges (`http://host/v1#model`, OpenAI-compatible, retries/backoff, key from env and never echoed; 7 tests against a local fake server, 128 total)
- promptlab v0.8: `--workers N` runs cases concurrently on a thread pool (order preserved, per-case error isolation, thread-safe CostMeter so budgets hold; 6 tests, 134 total)
- vecdb v0.1 (NEW): vector search from scratch — exact flat, IVF-Flat, IVF-PQ (residual PQ, ADC tables, exact re-rank), HNSW (paper's diversity heuristic, tombstones, resumable persistence), `Collection` with ids + metadata + Mongo-style filters pushed into the search, upsert/delete/compact, npz persistence; honest findings: HNSW wins on distance evaluations but loses wall-clock to one BLAS scan at n=20k in Python, PQ-only recall plateaus at 0.744, selective filters make graph walks catastrophic (exact scan of matching rows won at every selectivity; fallback threshold 0.15 from eval-count crossover); 76 tests, synthetic data
- vecdb v0.2: `SQ8Index` (uint8 scalar quantisation, 4x memory cut, optional exact refine; recall 0.98 -> 1.00 with refine=2 on synthetic data; memory win only, not speed; 7 tests, 85 total)
- vecdb v0.3: `HNSWIndex.repair()` (relink around tombstones, entry-point fix; recall unchanged on synthetic data, 25-76% fewer distance evals at 30-80% deleted; 4 tests, 89 total)
- vecdb v0.4: `HybridCollection` — BM25 + vector search with RRF or weighted fusion, filters on both sides, upsert/delete sync (8 tests, 97 total; fixtures only, no quality benchmark yet)
- vecdb v0.5: `parallel_search` threaded batch search (identical results to serial; ~1.4x on 2 cores for flat scans, none for HNSW; 4 tests, 101 total)
- vecdb v0.6: `OPQ` learned rotation before PQ (MSE -18% and recall@10 0.460 -> 0.503 on rotated synthetic data, tie on axis-aligned; 5 tests, 106 total; not yet in IVFIndex)
- vecdb v0.7: `IVFIndex(opq=True)` — learned rotation on residuals, persisted (MSE -5%, recall@10 0.539 -> 0.567 on synthetic data, single seed; 4 tests, 110 total)
- vecdb v0.9: hybrid fusion benchmark (`experiments/hybrid_eval.py`, dense vs BM25 vs RRF vs weighted; null result: fusion does not beat best single retriever on synthetic mixed queries; 2 tests, 118 total)
- promptlab: concurrent judge calls within one ensemble pair (--workers, order-preserving, serial-equivalence + overlap tests; 139 tests)
- docqa: `retrieval="hybrid"` (BM25 + char-trigram dense via vecdb, RRF); null result on all four eval sets, disclosed; 5 tests, 69 total
- tiny-transformer: `speculative_generate` (draft+verify sampling, residual resampling; greedy-equals-target, identical-draft and distribution-match tests; no KV cache so no wall-clock claims; 140 tests)
- tiny-transformer: `expected_tokens_per_call` (closed form) and `sweep_k` to compare measured tokens/target-call with the i.i.d.-acceptance prediction (3 tests, 143 total)
- tiny-transformer: `expected_speedup` / `best_k` wall-clock cost model for speculative decoding (speedup = tokens-per-call / (k*c+1); analytic, not measured; 4 tests)
- vecdb: SQ8Index persistence via save_index/load_index, incl. keep_raw and untrained (3 tests)
- vecdb: `HybridCollection.query_mmr` diversity re-ranking (lambda_=1 equals query; near-duplicate demotion test; 3 tests)
- vecdb: `fusion="zscore"` for HybridCollection.query (per-list standardisation instead of min-max; 4 tests; no quality claim, not benchmarked)
- vecdb: `alpha` now also weights RRF fusion (2*alpha dense / 2*(1-alpha) BM25; default 0.5 unchanged; 1 test, 137 total)
- vecdb: `HybridCollection.query_rerank` two-stage retrieval with a pluggable scorer and first-stage blend (4 tests, 141 total)
- vecdb: HybridCollection.query_terms phrases= (consecutive-token phrase constraint, 1 test)
- microgrid v0.1: numpy autodiff engine + gradcheck + optimizers (10 tests)
- microgrid: conv2d (stride/pad/bias) + max_pool2d with gradchecks, iterative backward, tiny CNN test (16 tests)
- microgrid: transformer blocks (LayerNorm, causal multi-head attention, GELU MLP, MiniGPT) with gradchecks + causality test (8 tests, 24 total)
- microgrid: AdamW (decoupled decay), global grad-norm clipping, warmup+cosine LR schedule, wired into lm.train (5 tests, 31 total)
- microgrid: numerical parity vs tiny-transformer (shared weights; logits, loss, all grads agree; 33 tests)
- microgrid: dropout op + Dropout module with train/eval (5 tests, 38 total)
- microgrid: leak-free held-out LM run on stdlib source (disjoint files; held-out loss 2.47 vs uniform 4.53; 39 tests)
- microgrid: KV-cache inference (exact vs autodiff forward and lm.generate; 12-33x faster than tape-based generation; 44 tests)
- microgrid: 3-seed held-out run + weight-decay ablation (null result, 2.568 vs 2.558 +- ~0.018)
- microgrid: int8-KV vs exact perplexity on held-out stdlib source (3 seeds, delta <2e-4 nats; cached_nll; 53 tests)
- microgrid: weight-only quantisation (8/6/4/3/2-bit, per-channel vs per-tensor; 4-bit per-channel costs ~0.02 nats; 60 tests)
- microgrid: group-wise weight scales (g8/g16; no gain at 4-bit, 3-bit loss +0.18 -> +0.11; 69 tests)
- microgrid: fake_quant_mse (per-channel MSE-optimal clipping; on synthetic Student-t(3) weights MSE drops 1.93->1.35 at 2-bit, 0.216->0.166 at 4-bit, ~0 at 6-bit; weight-error only, not task loss; 71 tests)
- specdec v0.1: speculative sampling, chi-square exactness test, acceptance/speedup experiment (6 tests)
- specdec v0.4: m-ary draft-tree verification (tree_generate), exactness test, tokens-per-target-call experiment (16 tests)
- specdec v0.5: cost-charged tree experiment (m=1 wins on near-uniform toy target; null result disclosed; 18 tests)
- specdec v0.6: peaky-target re-test of wide trees (m=2 wins only with poor drafts, up to ~6%; TemperedLM/MarkovLM; 19 tests)
- specdec v0.8: model-free prompt-lookup draft (exact; alpha 0.80 on periodic vs 0.04 unigram, worse on random; 27 tests)
- specdec: optimal_k(alpha, cost_ratio) closed-form-by-search helper + 4 tests (34 total)
- specdec: estimate_alpha / recommend_k offline calibration of draft length from sample contexts (+3 tests, 39 total)
- reviewkit: light intra-function taint (input/sys.argv/request.* -> SQL execute, shell, eval/exec; flow-insensitive, no inter-procedural; 88 tests)
- reviewkit: AST rules py-assert-tuple (always-true assert) and py-return-in-finally (90 tests)
- reviewkit: AST rule py-duplicate-dict-key (repeated constant keys in a dict literal; 91 tests)
- reviewkit: AST rule py-raise-notimplemented (`raise NotImplemented`; 92 tests)
- reviewkit: AST rule py-compare-none (`== None` / `!= None`; 93 tests)
- reviewkit: AST rule py-not-in-style (`not a in b` / `not a is b`; 94 tests)
- reviewkit: AST rule py-self-compare (name compared with itself; 95 tests)
- reviewkit: AST rule py-unreachable-code (statements after return/raise/continue/break; 96 tests)
- reviewkit: AST rule py-compare-bool (`== True` / `!= False`; 97 tests)
- reviewkit: AST rule py-duplicate-except (exception type caught twice in one try; 98 tests)
- reviewkit: AST rule py-fstring-no-placeholder (f-string with no placeholders; 99 tests)
- reviewkit: AST rule py-blocking-in-async (time.sleep/requests inside async def; 100 tests)
- reviewkit: AST rule py-useless-comparison (comparison as a bare statement; 101 tests)
- reviewkit: line rule py-mktemp (insecure tempfile.mktemp; 102 tests)
- docqa: GroundedGenerator, routed generative answers with a number/word grounding verifier and extractive fallback (10 tests)
- reviewkit: line rule py-insecure-random-secret (random used for tokens/passwords; 103 tests)
- visionlab: photometric (contrast) augmentation; fixes trained-family contrast, collapses to chance on an unseen contrast-about-mean shift (66 tests)
- reviewkit: py-ssl-unverified rule (104 tests)
- reviewkit: py-tar-extractall and py-chmod-world-writable rules (106 tests)
- reviewkit: py-flask-debug and py-xml-untrusted rules (108 tests)
- reviewkit: py-trust-remote-code rule (111 tests)
- reviewkit: py-dangerous-deserialization rule (112 tests)
- reviewkit: py-bind-all-interfaces rule (host="0.0.0.0" / bind((0.0.0.0,...)), 113 tests)
- reviewkit: line rule py-os-system (os.system/os.popen with non-constant command; 114 tests)
- reviewkit: py-log-secret rule (115 tests)
- reviewkit: py-hardcoded-tmp rule (fixed /tmp path in open(); 116 tests)
- reviewkit: py-cleartext-http rule (non-local http:// in requests/httpx calls; 117 tests)
- reviewkit: py-hardcoded-bearer rule (literal Bearer tokens; 118 tests)
- reviewkit: py-django-debug-true rule (119 tests)
- codesearch: `undocumented` command (public definitions lacking docstrings, ranked by name-based reference count; 3 tests)
- codesearch: `complexity` command (McCabe cyclomatic complexity per function via AST; dogfooded on this monorepo, top hit reviewkit ast_findings at 121; 3 tests)
- voicekit v0.1 (NEW): offline voice-command pipeline on synthetic formant speech — streaming VAD, MFCC, MLP vs DTW, open-set rejection, intent grammar, CLI, evaluation protocol (81 tests)
- bpekit v0.1: from-scratch byte-level BPE tokenizer (7 tests)
- bpekit: train_fast incremental trainer, identical merges, ~15x faster on 4.4 MB (11 tests)
- bpekit: GPT-2-style vocab.json/merges.txt export + validated import (round-trip identical encodings; 15 tests)
- bpekit: BPE-dropout stochastic encoding (lossless, seeded; 20 tests)
- bpekit: vocab-size sweep (held-out bytes/token vs vocab; 23 tests)
- datadoctor: `per_class_summary` (per-class flagged rates + top given->suggested confusions; 2 tests, 78 total)
- bpekit: cross-domain transfer matrix (prose/python/json; 29 tests)
- bpekit: `encode_with_offsets` (token -> char spans) and `chunk_by_tokens` (word-boundary, token-budgeted chunks with overlap; 35 tests)
- voicekit: `WakeGate` wake-word session gating (armed window, weak-wake rejection; 90 tests)
- reviewkit: AST rule py-catch-baseexception (`except BaseException` without re-raise; 120 tests)
- reviewkit: AST rule py-raise-notimplemented (`raise NotImplemented`; tests pass)
- voicekit: `Assistant` end-to-end streaming (audio -> wake gate -> intents, quiet-window command grouping; 97 tests)
- bpekit: `prune` drops merges a reference corpus never emits (keeping needed components; same token count on reference, 39 tests)
- voicekit: `merge_duplicates` collapses split same-word detections (93 tests with 3 new)
- voicekit: `margin_filter` drops ambiguous top-2-margin events (94 tests)
- bpekit: `StreamDecoder` / `stream_decode` UTF-8-safe incremental decoding (5 tests)
- bpekit: `extend` adds domain merges while keeping old ids stable (48 tests)
- reviewkit: AST rule py-utcnow (naive/deprecated `datetime.utcnow`; 121 tests)
- reviewkit: AST rule py-float-equality (`==`/`!=` vs non-integer float literal; 122 tests)
- bpekit: fertility metric (tokens/word, p95, single-token share, worst words; 3 tests, 51 total)
- bpekit: `truncate` token-budget truncation (head/tail/middle, UTF-8 safe; 57 tests)
- logsentry: seasonality-aware volume detector `detect_seasonal` (6 tests, v0.7)
- reviewkit: AST rule py-naive-now (`datetime.now()`/`today()` without tz; 123 tests)
- reviewkit: AST rule py-open-no-encoding (text-mode `open()` without encoding; 124 tests)
- bpekit: `compare` two tokenizers (bytes/token, ratio, boundary Jaccard; 3 tests, 64 total)
- datadoctor: class_balance + label_shift (TV distance, chi-square vs train; 5 tests, 86 total)
- reviewkit: AST rule py-urlopen-no-timeout (125 tests)
- bpekit: `pack` sequence packing into fixed-length rows with doc-start offsets (3 tests, 64 total)
- bpekit: `undertrained` lists rarely/never-emitted merged tokens (glitch-token candidates; 3 tests)
- reviewkit: AST-free rule py-cleartext-protocol (FTP/Telnet/plain SMTP; 126 tests)
- bpekit: `length_hist` token byte-length distribution + byte-fallback share (2 tests, 72 total)
- bpekit: `token_entropy` unigram entropy / bits-per-byte / efficiency (3 tests, 75 total)


## Next

1. rag-evals: (done hard benchmark) (done chunk sweep) -> next: dense embeddings beyond LSA
2. llm-eval-harness: (done) (done matrix) (done per-tag matrix) -> next: HTTP-model matrix specs
3. tiny-transformer follow-ups: (done v0.5, top-p, grad accumulation); (done clipping); (done dropout); (done weight tying); (done early stopping); (done repetition penalty); (done LR finder); (done label smoothing); (done per-position eval); (done sliding-window eval); (done lora-lab on its attention) (done RoPE) (done GQA/MQA) (done context extension: PI + NTK) -> next: a key-value recall benchmark with a larger model to test long-context use properly
4. agentflow follow-ups: (done shared memory) (done TTL) (done memory-aware Supervisor) (done failure fallback retries) (done circuit breaker) (done run metrics) (done Supervisor-in-DAG adapter) (done llm-router backend) (done persistent run log + resume) (done async orchestrator) (done async RunLog resume) -> next: async tool-call-level checkpointing
5. lora-lab: (done rank sweep) (done LoRA on tiny-transformer attention) (done adapter hub) (done QLoRA-style quantized base) -> (done LoftQ-style init) (done double quantization + bits/error frontier) -> next: paged/streamed adapter serving, or 2-bit (NF2) bases to see where LoftQ finally matters
6. logsentry: (done v0.6 Grafana dashboard)
7. ticket-triage: (done urgency head) (done active learning) (done multi-label tags) (done tag-aware routing) (done SLA-breach simulation) (done drift monitor: PSI/OOV/confidence, 61 tests) -> next: feed live router output into the simulator
8. llm-router: (done v0.1) -> (done latency tracking) (done semantic cache) (done learned complexity classifier) (done persistence)
9. promptlab: (done v0.1) -> v0.2 multi-sample done; (done GitHub Action + summary); (done LLM-judge checks + calibration) (done pairwise judge + position-swap bias audit) (done multi-judge ensembles + Fleiss kappa, v0.5) (done pairwise gate) (done ensemble gate) (done cost metering + budgets) (done HTTP-backed models/judges) (done concurrent cases, --workers) (done shared 429 cooldown across workers) (done concurrent ensemble judge calls)
10. docqa: (done v0.1) -> (done paraphrase) (done broad benchmark) (done yes/no fixes) (done agentkit integration) (done agentflow worker) (done HTTP service) (done cache + metrics) (done typo-tolerant spelling) (done hybrid retrieval, null result) (done docqa on llm-router: grounded generation) next: answer-stage semantic scoring
11. autojob: (done v0.1 — profile validation, posting parse, transparent match scoring, facts-only cover-letter drafts, review notes, JSONL tracker; never submits; 49 tests) (done v0.2 — status history + funnel analytics, 66 tests) (done v0.3 — interview-prep packets, 77 tests) (done v0.4 — follow-up reminders with facts-only draft emails, 93 tests) (done v0.5 — tailored resume sections: skills matrix, lead-with order, role ranking, summary ingredients; 107 tests) (done v0.6 — weekly digest + gap leaderboard, 123 tests) (done v0.7 — gap-closing plan, 143 tests) (done v0.8 — posting ranking, 162 tests) (done v0.9 — salary analysis, 189 tests) (done v0.10 — offer comparison, 219 tests) (done v0.11 — interview rounds log, 254 tests) (done v0.12 — calendar export, 283 tests) (done v0.13 — contacts log, 316 tests) (done v0.14 — activity timeline, 340 tests) (done v0.15 — posting closing dates, 374 tests)
12. visionlab follow-ups: (done GAP head vs dense under shift) (done batch-norm) (done certified robustness via IBP) -> next: contrast/photometric augmentation to fix the GAP head's contrast brittleness, tighter bounds (CROWN / zonotope) to certify the FGSM-trained net, IBP-compatible BatchNorm
13. reviewkit follow-ups: a fresh held-out set (the current one is now tuned-on); (done moved-code detection, 80 tests); (done light intra-function taint for SQL/shell/eval, 88 tests); run the Action on a real PR in a scratch repo; docqa/ragkit-style repo-context retrieval for the LLM pass; JS/TS AST via a parser
14. codesearch follow-ups: optional neural-embedding backend as an extra (compare against BM25F on the same queries), a third blind query set written by someone other than the author, call-graph expansion (boost callers/callees of top hits), JS/TS via a parser, (done `watch` re-index, v0.x), (done JSON-lines `serve`, 69 tests), (done `calls` name-based callers/callees, 74 tests), (done `hotspots`: complexity x references, 87 tests), (done `longest`: functions by line count, 90 tests)
15. datadoctor follow-ups: (done audit diff + CI regression gate) (done temperature scaling, mixed results) a real noisy-label dataset benchmark (needs a download-free source), (done regression targets) (done ordinal targets) (done co-occurrence stacking, null result on detection F1) (done feature_drift: KS + PSI, 91 tests) object-detection style boxes
16. vecdb follow-ups: (done flat SQ8) (done HNSW tombstone repair) (done SQ8 in IVF cells + persistence, v0.8), (done OPQ class) (done OPQ inside IVFIndex), (done multi-threaded batch search), (done hybrid BM25+vector core) (done HybridCollection save/load, v0.10) (done HybridCollection.explain: per-hit dense/BM25 ranks, v0.11) (done HybridCollection.query_batch) (done configurable + persisted BM25 k1/b) (done HybridCollection.query_terms must/must_not lexical constraints) (done Collection.find_duplicates near-duplicate pairs) wire into ragkit/docqa and benchmark fusion quality, real-embedding benchmark
17. microgrid follow-ups: (done conv2d/pooling) (done iterative backward) (done transformer blocks) (done char-LM training + generation on the corpus, memorisation disclosed) (done AdamW/clip/cosine schedule) (done parity vs tiny-transformer) (done leak-free stdlib-source corpus run) (done KV-cache generation) (done 3-seed + wd ablation) (done int8 KV cache + top-k/top-p sampling) (done int8-KV perplexity) (done weight-only quantisation) (done GPTQ-style rounding vs RTN) (done group-wise scales: helps at <=3 bits only) (done sequential Hessian re-collection: helps most at 2-3 bits) -> next: a new project
18. specdec follow-ups: (done peaky target re-test) (done adaptive k) (done multi-draft recursive rejection at one position) (done m-ary tree along accepted path) (done microgrid real draft/target pair, v0.7: c=0.49 limits gain) (done backoff lookup draft: ties best fixed n, no gain) (done batched verification API, v0.9: exact, no wall-clock win without KV cache) (done CLI: calibrate+benchmark on any file) full branching-tree verification, KV-cached verification
19. voicekit follow-ups: CNN word classifier on microgrid, (done per-word thresholds from validation data, 84 tests) (done benchmark vs global threshold: null result, F1 0.953/0.957/0.955) (done WakeGate event gating) (done WakeGate max_session_s cap, 2 tests) (done `smooth_events` posterior smoothing, 3 tests) better VAD below 5 dB, a real-speech benchmark from a download-free source
