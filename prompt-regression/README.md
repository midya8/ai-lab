# prompt-regression

Treat prompts like code: version them, compare two versions with proper paired statistics, and fail CI when a prompt change breaks a case that used to pass. Standard library only.

I built it because "I rewrote the prompt and it feels better" kept being the whole argument. On a 20-case test set a 5-point swing is usually noise, and a change that raises the average can still quietly break things.

The bundled `RuleModel` is a deterministic fake LLM so everything runs offline. Point `--model cmd:<command>` or `--model http://host/v1#model` at a real one.

```bash
cd prompt-regression
bash examples/demo.sh                      # end-to-end run in a temp registry
python -m unittest discover -s tests

python -m promptreg --root prompts commit triage --file triage.txt -m "add categories"
python -m promptreg --root prompts compare triage v1 v3 --cases cases.jsonl --report ab.md
python -m promptreg --root prompts baseline triage --ref prod --cases cases.jsonl --out baseline.json
python -m promptreg --root prompts check triage --ref latest --cases cases.jsonl --baseline baseline.json
```

`check` exits 1 if any previously passing case now fails.

## Demo result

Three versions of a ticket-triage prompt on 60 generated tickets (fake model, so these numbers just show the plumbing):

| version | change | pass rate |
|---|---|---|
| v1 | "Classify the support ticket." | 58.3% |
| v2 | + category list, one-word answer | 90.0% |
| v3 | + delimiters, worked example, JSON output | 98.3% |

v1 vs v2 is a clear win (exact McNemar p = 0.0005, 24 cases fixed, 5 broken). Gating v3 against a v2 baseline still fails, because one case regressed. That's the point of the gate.

## What else is in here

- Test cases are JSONL with checks like `contains`, `regex`, `max_words`, `json_field`.
- `judge` checks use a second model to grade against a rubric. `calibrate` measures how well the judge agrees with human labels (agreement, Cohen's kappa, Spearman, leniency). The bundled lexical judge gets kappa 0.44 on 32 labels I wrote, which is about what you'd expect.
- `pairwise` judges each pair twice with the order swapped, so a judge that just prefers the first slot shows up as inconsistency instead of as a result. `pair-audit` checks a pairwise judge against human preferences.
- Multi-judge majority votes with Fleiss' kappa, `--samples N` for non-deterministic models, cost metering with a hard `--budget`, `--workers N` with a shared 429 cooldown.
- `action/action.yml` is a composite GitHub Action that runs `check` and writes a summary (`uses: ./prompt-regression/action`).

The HTTP adapter has only been tested against a local fake server. A real LLM judge needs its own calibration with a bigger, independently labelled set than the examples here.
