"""Development benchmark. The rules were tuned against these cases (see README); the
held-out set in heldout_cases.py was written afterwards and run once before any further change.

Token-shaped strings are assembled from pieces so this file itself never contains a
literal secret (and GitHub push protection stays quiet)."""

SUBS = {
    "@@AWS@@": "AKIA" + "IOSFODNN7EXAMPLE",
    "@@GH@@": "ghp_" + "a1B2c3D4e5F6g7H8i9J0" * 2,
    "@@SK@@": "sk-" + "proj" + "Qw3rTy9uIoP0aSdFgHjK",
    "@@PEM@@": "-----BEGIN RSA " + "PRIVATE KEY-----",
}


def T(s):
    for k, v in SUBS.items():
        s = s.replace(k, v)
    return s


CASES = [
    {"name": "eval-and-exec", "files": {"app/calc.py": (None, T('''import ast


def evaluate(expr):
    return eval(expr)  # EXPECT: py-eval


def run(code):
    exec(code)  # EXPECT: py-eval


def safe(expr):
    return ast.literal_eval(expr)
'''))}},
    {"name": "eval-lookalikes-are-clean", "files": {"ml/train.py": (None, T('''import torch


def fit(model, loader):
    model.eval()
    # never call eval(user_input) here
    msg = "do not eval(x) anything"
    doc = """
    Example: exec(cmd) is dangerous.
    """
    return model, msg, doc
'''))}},
    {"name": "shell-and-pickle", "files": {"tools/run.py": (None, T('''import pickle
import subprocess


def go(cmd, blob):
    subprocess.run(cmd, shell=True)  # EXPECT: py-shell-true
    return pickle.loads(blob)  # EXPECT: py-unsafe-deserialize


def ok(arg):
    subprocess.run(["ls", arg], check=True)
'''))}},
    {"name": "torch-yaml-multiline", "files": {"ml/io.py": (None, T('''import torch
import yaml


def load_bad(p):
    return torch.load(p)  # EXPECT: py-torch-load


def load_good(p):
    return torch.load(
        p,
        map_location="cpu",
        weights_only=True,
    )


def cfg_bad(f):
    return yaml.load(f)  # EXPECT: py-yaml-load


def cfg_good(f):
    return yaml.safe_load(f)


def cfg_loader(f):
    return yaml.load(f, Loader=yaml.SafeLoader)
'''))}},
    {"name": "tls-hash-debug", "files": {"svc/net.py": (None, T('''import hashlib
import requests


def fetch(u):
    return requests.get(u, verify=False, timeout=5)  # EXPECT: py-tls-verify-off


def digest(b):
    return hashlib.md5(b).hexdigest()  # EXPECT: py-weak-hash


def cache_key(b):
    return hashlib.md5(b, usedforsecurity=False).hexdigest()


def debug():
    breakpoint()  # EXPECT: py-debug-leftover
'''))}},
    {"name": "sql-injection", "files": {"db/queries.py": (None, T('''def by_name(cur, name):
    cur.execute("SELECT * FROM users WHERE name = '%s'" % name)  # EXPECT: py-sql-injection
    cur.execute(f"DELETE FROM users WHERE name = '{name}'")  # EXPECT: py-sql-injection
    cur.execute("SELECT * FROM users WHERE name = %s", (name,))
    q = f"select id from t where x = {name}"  # EXPECT: py-sql-injection
    return cur.fetchall(), q


def fine(cur):
    cur.execute("SELECT 1")
    msg = f"selected {cur} rows"
    return msg
'''))}},
    {"name": "secrets", "files": {"config/settings.py": (None, T('''import os

AWS_KEY = "@@AWS@@"  # EXPECT: secret-token
GITHUB = "@@GH@@"  # EXPECT: secret-token
LLM_KEY = "@@SK@@"  # EXPECT: secret-token
PEM = """@@PEM@@"""  # EXPECT: secret-token
DB_PASSWORD = "hunter2-hunter2"  # EXPECT: secret-assignment
api_key = "k9X3mQ7vL2pR8sT4"  # EXPECT: secret-assignment
FROM_ENV = os.environ["DB_PASSWORD"]
PLACEHOLDER = "your-api-key-here"
password_hint = "changeme-please"
SHORT_TOKEN = "abc"
TOKEN_URL = "https://example.com/oauth/token"
'''))}},
    {"name": "secrets-in-tests-are-lenient", "files": {"tests/test_auth.py": (None, T('''def test_login():
    password = "correct-horse-battery"
    key = "@@AWS@@"  # EXPECT: secret-token
    assert password and key
'''))}},
    {"name": "mutable-defaults", "files": {"lib/util.py": (None, T('''from collections import defaultdict


def collect(item, bucket=[]):  # EXPECT: py-mutable-default
    bucket.append(item)
    return bucket


def index(rows, table=defaultdict(list)):  # EXPECT: py-mutable-default
    return table


def good(item, bucket=None, size=3, name="x", flags=()):
    bucket = bucket or []
    return bucket
'''))}},
    {"name": "exceptions", "files": {"lib/safe.py": (None, T('''import logging

log = logging.getLogger(__name__)


def a(f):
    try:
        f()
    except:  # EXPECT: py-bare-except
        log.exception("boom")


def b(f):
    try:
        f()
    except ValueError:  # EXPECT: py-swallowed-exception
        pass


def c(f):
    try:
        f()
    except:  # EXPECT: py-bare-except, py-swallowed-exception
        pass


def d(f):
    try:
        f()
    except KeyError:
        log.warning("missing")
'''))}},
    {"name": "requests-timeout", "files": {"svc/client.py": (None, T('''import requests


def a(u):
    return requests.get(u)  # EXPECT: py-requests-no-timeout


def b(u, d):
    return requests.post(u, json=d, timeout=10)


def c(u, **kw):
    return requests.get(u, **kw)


def d(session, u):
    return session.get(u)
'''))}},
    {"name": "is-literal", "files": {"lib/cmp.py": (None, T('''def f(x, y):
    if x is "yes":  # EXPECT: py-is-literal
        return 1
    if y is not 0:  # EXPECT: py-is-literal
        return 2
    if x is None or y is True:
        return 3
    return 4
'''))}},
    {"name": "complexity", "files": {"lib/big.py": (None, T('''def tangled(a, b, c):  # EXPECT: py-complexity
    out = 0
    if a:
        out += 1
    if b:
        out += 2
    if c:
        out += 3
    for i in range(a):
        if i % 2 and i % 3:
            out += i
        elif i % 5 or i % 7:
            out -= i
    while b > 0:
        b -= 1
        if b == 4:
            break
    return out


def simple(a):
    return a + 1
'''))}},
    {"name": "preexisting-risk-not-flagged", "files": {"legacy/old.py": (
        T('''import pickle


def load(blob):
    return pickle.loads(blob)


def run(x):
    return eval(x)


def tidy(x):
    return x
'''),
        T('''import pickle


def load(blob):
    return pickle.loads(blob)


def run(x):
    return eval(x)


def tidy(x):
    return x + 1


def new_feature(y):
    return y * 2
'''))}},
    {"name": "new-risk-next-to-old-risk", "files": {"legacy/mix.py": (
        T('''def run(x):
    return eval(x)
'''),
        T('''def run(x):
    return eval(x)


def run2(x):
    return eval(x)  # EXPECT: py-eval
'''))}},
    {"name": "javascript", "files": {"web/app.js": (None, T('''const el = document.getElementById("out");
el.innerHTML = userHtml;  // EXPECT: js-inner-html
const v = eval(input);  // EXPECT: js-eval
const f = new Function("a", "return a");  // EXPECT: js-eval
// eval(this) is only mentioned in a comment
el.textContent = "eval(x) in a string";
if (a === b) { run(); }
'''))}},
    {"name": "merge-conflict", "files": {"lib/conflict.py": (None, T('''def f():
    pass
<<<<<<< HEAD  # EXPECT: merge-conflict, py-syntax-error
x = 1
======= # EXPECT: merge-conflict
x = 2
>>>>>>> feature  # EXPECT: merge-conflict
'''))}},
    {"name": "clean-ml-code", "files": {"ml/model.py": (None, T('''import numpy as np
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: tuple = (64, 64)):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.body = nn.Sequential(*layers, nn.Linear(d, d_out))

    def forward(self, x):
        return self.body(x)


def evaluate(model, xs, ys):
    model.eval()
    with torch.no_grad():
        pred = model(xs).argmax(-1)
    return float((pred == ys).float().mean())


def load(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    return state


def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
'''))}},
    {"name": "method-named-eval-found-by-dogfooding", "files": {"nn/module.py": (None, T('''class Module:
    def eval(self):
        self.training = False
        return self

    def exec(self, plan):
        return plan

    def go(self, code):
        return exec(code)  # EXPECT: py-eval
'''))}},
    {"name": "inline-suppression", "files": {"tools/trusted.py": (None, T('''def run(expr):
    return eval(expr)  # reviewkit: ignore[py-eval]
'''))}},
]
