"""Authoring helper: writes dev/held-out query files with labels checked against the live index.

Labels are written as display ids (``agentflow/agentflow/core.py::X``, with ``/src/`` elided) and
resolved to real chunk ids so a typo fails loudly instead of silently scoring as a miss.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from codesearch.store import sync  # noqa: E402

INCLUDE = ["agentflow", "agentkit", "docqa", "guardkit", "llm-router", "promptlab", "ragkit", "reviewkit"]
AF = "agentflow/agentflow/"
LR = "llm-router/llmrouter/"
PL = "promptlab/promptlab/"
GK = "guardkit/guardkit/"
DQ = "docqa/docqa/"
AK = "agentkit/agentkit/"
RK = "ragkit/ragkit/"
RV = "reviewkit/reviewkit/"

DEV = [
    ("per-model failure tracker that stops sending traffic after repeated errors and probes again after a cooldown",
     [AF + "extras.py::CircuitBreaker", LR + "breaker.py::CircuitBreaker"]),
    ("split a document into overlapping pieces on sentence boundaries", [RK + "chunk.py::split_text"]),
    ("verify a credit card number checksum", [GK + "detectors.py::luhn_ok"]),
    ("exact p-value for paired binary outcomes", [PL + "stats.py::mcnemar_exact"]),
    ("confidence interval for a proportion with small samples", [PL + "stats.py::wilson"]),
    ("write the cache to disk atomically and reload it after a restart",
     [LR + "semantic.py::SemanticCache.save", LR + "semantic.py::SemanticCache.load"]),
    ("put original values back after redaction", [GK + "redact.py::RedactionVault.restore"]),
    ("order tasks so dependencies run first and detect circular dependencies", [AF + "core.py::Orchestrator.plan"]),
    ("edit distance between two words allowing transposition", [DQ + "spell.py::osa_distance"]),
    ("turn a grader model reply into a score and reason, rejecting malformed output", [PL + "judge.py::parse_judge_reply"]),
    ("roughly how many tokens will this prompt use", [LR + "models.py::estimate_tokens"]),
    ("journal of finished tasks so an interrupted workflow can continue",
     [AF + "runlog.py::RunLog", AF + "runlog.py::RunLog.completed", AF + "runlog.py::RunLog.record"]),
    ("prevent reading files outside an allowed directory", [AK + "tools.py::make_read_file"]),
    ("evaluate arithmetic safely without calling eval", [AK + "tools.py::calculator", AK + "tools.py::_eval_node"]),
    ("load a policy file and fail if it is invalid", [GK + "policy.py::load_policy"]),
    ("avoid posting a review comment that was already posted",
     [RV + "github.py::existing_fingerprints", RV + "github.py::post_review"]),
    ("percentile of latency measurements", [DQ + "metrics.py::percentile"]),
    ("correlation between two rankings", [PL + "stats.py::spearman"]),
    ("Prometheus text exposition of request counters and histograms", [DQ + "metrics.py::Metrics.prometheus"]),
    ("stable hash of a prompt template that ignores trailing whitespace", [PL + "template.py::Template.digest"]),
    ("cache key that ignores capitalization and extra spaces", [LR + "cache.py::cache_key", DQ + "cache.py::normalize"]),
    ("mask phone numbers and emails found in text with placeholder tokens",
     [GK + "redact.py::redact_spans", GK + "detectors.py::PiiSecretScanner.scan"]),
    ("drop expired entries from agent memory", [AF + "memory.py::SharedMemory.purge"]),
    ("ask the grader several times and take the conservative middle score", [PL + "judge.py::judge_score"]),
    ("decide which worker should handle a task", [AF + "extras.py::Supervisor.route"]),
    ("neutralise untrusted text so it cannot close the prompt section",
     [PL + "judge.py::_defang", PL + "pairwise.py::_defang", RV + "llm.py::neutralize"]),
]

HELD = [
    ("choose the cheapest model that meets the quality needed for this prompt",
     [LR + "router.py::Router.plan", LR + "router.py::Router._order"]),
    ("cosine similarity between hashed text vectors", [LR + "semantic.py::cosine", LR + "semantic.py::embed"]),
    ("stop spending when the budget window is used up and degrade to cheaper models",
     [LR + "budget.py::Budget.degraded", LR + "budget.py::Budget.can_afford", LR + "budget.py::Budget.pressure"]),
    ("agreement statistic between two raters that corrects for chance", [PL + "stats.py::cohen_kappa"]),
    ("bootstrap resampling of paired differences", [PL + "stats.py::paired_bootstrap"]),
    ("suggest the nearest known vocabulary word for a misspelling",
     [DQ + "spell.py::SpellCorrector.suggest", DQ + "spell.py::SpellCorrector.correct"]),
    ("convert unified diff text into per-file hunks with line numbers", [RV + "diff.py::parse_diff"]),
    ("call an HTTP chat completions endpoint and retry when rate limited",
     [RV + "llm.py::OpenAICompatProvider.complete", AK + "providers.py::OpenAICompatibleProvider.complete",
      RK + "answer.py::OpenAICompatibleAnswerer.answer"]),
    ("fraction of attacks the guardrail caught", [GK + "eval.py::EvalReport.attack_recall"]),
    ("markdown summary of the regression gate for a CI job page", [PL + "compare.py::render_check_markdown"]),
    ("sliding window of observed response times used to correct latency estimates",
     [LR + "latency.py::LatencyTracker.factor", LR + "latency.py::LatencyTracker.record"]),
    ("fake model that fails at random for tests", [LR + "providers.py::SimProvider", LR + "providers.py::SimProvider.complete"]),
    ("decide whether the answer is supported by the document or abstain",
     [DQ + "qa.py::DocQA.ask", DQ + "qa.py::DocQA.coverage", DQ + "qa.py::DocQA._ask"]),
    ("limit the number of steps an agent may take and stop when it repeats the same call",
     [AK + "runner.py::AgentRunner.run", AK + "model.py::ToolCall.canonical", AK + "runner.py::AgentRunner._check_call"]),
    ("time-to-live expiry check on stored entries",
     [LR + "semantic.py::SemanticCache._expired", AF + "memory.py::SharedMemory._expired"]),
    ("line numbers of added lines in a patch", [RV + "diff.py::FileDiff.added", RV + "diff.py::FileDiff.visible"]),
    ("count the decision points of a function", [RV + "rules.py::cyclomatic"]),
    ("substitute variables in a template and fail when one is missing", [PL + "template.py::Template.render"]),
    ("fingerprint of a finding that survives when code moves to a different line", [RV + "rules.py::Finding.fp"]),
    ("check tool call arguments against a JSON schema", [AK + "model.py::Tool.validate_args"]),
    ("drop reviewer comments that point at lines that do not exist", [RV + "llm.py::parse_reply"]),
    ("wrap a text function so that input and output are both checked", [GK + "guard.py::Guard.wrap"]),
    ("summarise past workflow runs one line each", [AF + "runlog.py::RunLog.runs", AF + "runlog.py::RunLog.format_runs"]),
    ("critical path and parallelism computed from task timings", [AF + "metrics.py::compute_metrics", AF + "metrics.py::RunMetrics"]),
    ("read a jsonl file of labelled items skipping comment lines",
     [PL + "runner.py::load_cases", PL + "judge.py::load_calibration_items", PL + "pairwise.py::load_pair_items",
      GK + "eval.py::load_cases"]),
    ("attach a label to a stored prompt version and look it up later",
     [PL + "registry.py::Registry.tag", PL + "registry.py::Registry.resolve"]),
]


def main(root: Path) -> None:
    chunks, _ = sync(root, None, INCLUDE)
    lookup = {c.id.replace("/src/", "/"): c.id for c in chunks}
    for name, qs in (("dev", DEV), ("heldout", HELD)):
        out = []
        for q, rel in qs:
            bad = [r for r in rel if r not in lookup]
            if bad:
                raise SystemExit(f"{name}: unknown labels for {q!r}: {bad}")
            out.append({"query": q, "relevant": [lookup[r] for r in rel]})
        (HERE / f"queries_{name}.json").write_text(json.dumps(
            {"root": "../..", "include": INCLUDE, "queries": out}, indent=1) + "\n")
        print(name, len(out))


if __name__ == "__main__":
    main(HERE.parent.parent)
