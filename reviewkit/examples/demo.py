"""End-to-end demo with no network: a sample PR, the rule pass, a scripted 'model', the review payload.

    PYTHONPATH=src python examples/demo.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reviewkit.bench import make_diff  # noqa: E402
from reviewkit.github import build_review  # noqa: E402
from reviewkit.llm import ScriptedProvider  # noqa: E402
from reviewkit.review import Reviewer, render_summary  # noqa: E402

BEFORE = '''import json


def load_config(path):
    with open(path) as f:
        return json.load(f)
'''
AFTER = '''import json
import subprocess

import requests


def load_config(path):
    with open(path) as f:
        return json.load(f)


def fetch_scores(url, cache={}):
    resp = requests.get(url)
    data = resp.json()
    cache[url] = data
    return sum(data["scores"]) / len(data["scores"])


def convert(name):
    subprocess.run("ffmpeg -i " + name + " out.mp4", shell=True)
'''

# What a model might say: one comment duplicates a rule, one is new, one points at a line that is not in the PR.
MODEL_REPLY = json.dumps({"comments": [
    {"line": 16, "severity": "medium", "message": "ZeroDivisionError when data['scores'] is empty; guard the division."},
    {"line": 20, "severity": "high", "message": "shell=True with a concatenated filename is command injection."},
    {"line": 9, "severity": "high", "message": "This line was not changed by the PR (the validator drops it)."},
]})

with tempfile.TemporaryDirectory() as root:
    with open(os.path.join(root, "scores.py"), "w") as f:
        f.write(AFTER)
    diff = make_diff({"scores.py": (BEFORE, AFTER)})
    result = Reviewer(root=root, provider=ScriptedProvider([MODEL_REPLY])).review(diff)

print(render_summary(result))
print()
print("LLM stats:", result.stats)
print()
print(json.dumps(build_review(result)["comments"][0], indent=2))
