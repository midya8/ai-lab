import json
import re


def _norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.\-]", "", s.lower())).strip()


def exact(out, case):
    return float(_norm(out) == _norm(case["expected"]))


def contains(out, case):
    return float(_norm(case["expected"]) in _norm(out))


def regex(out, case):
    return float(re.search(case["expected"], out) is not None)


def numeric(out, case):
    nums = re.findall(r"-?\d+(?:\.\d+)?", out)
    if not nums:
        return 0.0
    return float(abs(float(nums[-1]) - float(case["expected"])) <= case.get("tol", 1e-6))


def json_keys(out, case):
    try:
        obj = json.loads(out)
    except ValueError:
        return 0.0
    return float(isinstance(obj, dict) and all(k in obj for k in case["expected"]))


def token_f1(out, case):
    p, g = _norm(out).split(), _norm(case["expected"]).split()
    if not p or not g:
        return float(p == g)
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


SCORERS = {f.__name__: f for f in (exact, contains, regex, numeric, json_keys, token_f1)}


def score(name, out, case):
    if name not in SCORERS:
        raise KeyError(f"unknown scorer {name!r}; have {sorted(SCORERS)}")
    return SCORERS[name](out, case)


def judge_scorer(judge_backend, rubric="Is the answer correct and helpful?"):
    """Build an LLM-as-judge scorer: the judge replies with a 1-5 rating, mapped to 0..1."""
    import re

    def _judge(output, case):
        prompt = (f"{rubric}\nReference: {case.get('expected', '')}\nAnswer: {output}\n"
                  "Reply with a single integer rating from 1 to 5.")
        m = re.search(r"[1-5]", str(judge_backend.generate(prompt)))
        return (int(m.group()) - 1) / 4 if m else 0.0

    return _judge
