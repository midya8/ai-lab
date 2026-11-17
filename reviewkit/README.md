# reviewkit

A pull-request review bot. A deterministic rule pass catches what rules can catch reliably. An optional LLM pass adds judgement, and everything the model says is checked against the diff before it gets posted. Runs offline by default.

```bash
cd reviewkit
pip install -r requirements.txt
git diff main...HEAD | PYTHONPATH=src python -m reviewkit review --diff - --root . --fail-on high
PYTHONPATH=src python -m reviewkit rules
PYTHONPATH=src python -m reviewkit bench bench/dev_cases.py
PYTHONPATH=src python examples/demo.py
PYTHONPATH=src python -m pytest -q
```

How it works:

- It only reports on lines the PR added. Code that was already there, or just moved, isn't blamed on the author.
- About 25 rules: `eval`/`exec`, `shell=True`, pickle and `torch.load` without `weights_only`, `yaml.load`, `verify=False`, SQL built from strings, committed keys and tokens, mutable defaults, swallowed exceptions, `requests` without a timeout, debuggers left in, merge markers, and a few JS ones. There's also simple per-function taint tracking from `input()`, `sys.argv` and request data into SQL, shell and eval.
- Python rules run on a view of the code with strings and comments blanked out, so `model.eval()` or `# exec(cmd)` don't trigger anything.
- A leaked key is never repeated back into the PR thread.
- The LLM reply has to be valid JSON, every comment has to point at a line the PR actually added, severity is capped, and @mentions are defused so an injected diff can't ping people.
- Comments carry a fingerprint, so pushing again only posts new findings.

## How well it works

I wrote a dev set and then, after freezing the rules, a held-out set:

| set | precision | recall |
|---|---|---|
| dev (tuned on) | 1.000 | 1.000 |
| held-out, first run | 0.762 | 0.667 |
| held-out, after fixing real misses | 0.923 | 1.000 |

The held-out run found real gaps (`from pickle import loads`, `timeout=None`, `idx is -1`, and others), which I then fixed. So the last row isn't held-out anymore, and I need a fresh set to say anything new.

## Limitations

Rules are syntactic, and the taint tracking stays within a single function. JS/TS only gets a few line rules. `action.yml` hasn't been run on a real PR yet, only against a fake GitHub API in tests. And the model pass is only as good as the model. reviewkit limits what a bad reply can do, not whether it's right.
