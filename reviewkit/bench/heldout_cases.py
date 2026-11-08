"""Held-out benchmark: written AFTER the rules were frozen against dev_cases.py and run once
before any rule change. Several cases are deliberate probes of variations the rules were
not tuned for (a function imported by name, `timeout=None`, `.format()` into a variable...).
Expectations mark what a careful human reviewer would want flagged, whether or not the
rules can do it, so the misses are real misses."""
import importlib.util as _u
import os as _os

_spec = _u.spec_from_file_location("_dev_cases", _os.path.join(_os.path.dirname(__file__), "dev_cases.py"))
_m = _u.module_from_spec(_spec)
_spec.loader.exec_module(_m)
T = _m.T

CASES = [
    {"name": "fastapi-llm-service", "files": {"api/main.py": (None, T('''import os
import subprocess

import requests
from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
client = OpenAI(api_key="@@SK@@")  # EXPECT: secret-token


@app.post("/summarize")
def summarize(payload: dict):
    r = requests.post("https://api.example.com/v1", json=payload)  # EXPECT: py-requests-no-timeout
    return r.json()


@app.post("/convert")
def convert(name: str):
    subprocess.Popen("ffmpeg -i " + name + " out.mp4", shell = True)  # EXPECT: py-shell-true
    return {"ok": True}


@app.get("/health")
def health():
    return {"key_set": bool(os.environ.get("OPENAI_API_KEY"))}
'''))}},
    {"name": "named-imports-probe", "files": {"util/ser.py": (None, T('''from pickle import loads
import yaml


def parse(blob):
    return loads(blob)  # EXPECT: py-unsafe-deserialize


def conf(f):
    return yaml.load(f, Loader=yaml.SafeLoader)
'''))}},
    {"name": "sql-variants", "files": {"db/repo.py": (None, T('''def by_id(cur, i):
    cur.execute("SELECT * FROM t WHERE id=" + str(i))  # EXPECT: py-sql-injection
    cur.execute(
        f"""UPDATE accounts SET balance = {i} WHERE id = 1"""  # EXPECT: py-sql-injection
    )


def indirect(cur, a):
    sql = "SELECT * FROM t WHERE a = {}".format(a)  # EXPECT: py-sql-injection
    cur.execute(sql)


def not_sql(config, x):
    return f"Please update config set to {x} before continuing"


def parametrised(cur, ids):
    marks = ",".join(["%s"] * len(ids))
    cur.execute("SELECT * FROM t WHERE id IN (" + marks + ")", ids)
'''))}},
    {"name": "requests-edge", "files": {"svc/probe.py": (None, T('''import requests


def a(u):
    return requests.get(u, timeout=None)  # EXPECT: py-requests-no-timeout


def b(u):
    resp = requests.head(  # EXPECT: py-requests-no-timeout
        u,
        allow_redirects=True,
    )
    return resp


def c(u):
    return requests.get(u, timeout=(3, 10))
'''))}},
    {"name": "config-files-with-secrets", "files": {
        "deploy/app.yaml": (None, T('''service: demo
database:
  password: "s3cr3t-passw0rd-prod"  # EXPECT: secret-assignment
  host: db.internal
''')),
        ".env.example": (None, T('''API_KEY="your-api-key"
DB_PASSWORD="changeme"
SLACK=@@SK@@  # EXPECT: secret-token
''')),
        "notes/rotate.md": (None, T('''Rotate this key: @@AWS@@  # EXPECT: secret-token
''')),
    }},
    {"name": "dict-style-credentials", "files": {"ml/cfg.py": (None, T('''CONFIG = {
    "model": "gpt-x",
    "password": "tr0ub4dor-and-3",  # EXPECT: secret-assignment
    "token_limit": 4096,
    "api_key": os.environ["KEY"],
    "secret": get_secret("kms/alias"),
}
'''))}},
    {"name": "complexity-boundary", "files": {"lib/branches.py": (None, T('''def ten(a):
    r = 0
    if a == 1:
        r += 1
    if a == 2:
        r += 1
    if a == 3:
        r += 1
    if a == 4:
        r += 1
    if a == 5:
        r += 1
    if a == 6:
        r += 1
    if a == 7:
        r += 1
    if a == 8:
        r += 1
    if a == 9:
        r += 1
    return r


def eleven(a):  # EXPECT: py-complexity
    r = 0
    if a == 1:
        r += 1
    if a == 2:
        r += 1
    if a == 3:
        r += 1
    if a == 4:
        r += 1
    if a == 5:
        r += 1
    if a == 6:
        r += 1
    if a == 7:
        r += 1
    if a == 8:
        r += 1
    if a == 9:
        r += 1
    if a == 10:
        r += 1
    return r
'''))}},
    {"name": "mutable-default-variants", "files": {"lib/defaults.py": (None, T('''class Agent:
    def __init__(self, tools=[], *, memory={}):  # EXPECT: py-mutable-default, py-mutable-default
        self.tools, self.memory = tools, memory

    async def step(self, history=list()):  # EXPECT: py-mutable-default
        return history


add = lambda x, acc=[]: acc + [x]  # EXPECT: py-mutable-default
'''))}},
    {"name": "is-literal-variants", "files": {"lib/idx.py": (None, T('''def find(xs, v):
    idx = xs.index(v) if v in xs else -1
    if idx is -1:  # EXPECT: py-is-literal
        return None
    if v is "":  # EXPECT: py-is-literal
        return None
    if v is not None and idx is not Ellipsis:
        return idx
'''))}},
    {"name": "exception-variants", "files": {"lib/handlers.py": (None, T('''import contextlib


def a(f):
    try:
        f()
    except Exception:  # EXPECT: py-swallowed-exception
        ...


def b(f):
    with contextlib.suppress(Exception):
        f()
    try:
        f()
    except (ValueError, KeyError) as e:
        raise RuntimeError("bad") from e


def c(f):
    for i in range(3):
        try:
            return f(i)
        except TimeoutError:
            continue
'''))}},
    {"name": "js-variants", "files": {"web/ui.tsx": (None, T('''export function Card({ html }: { html: string }) {
  return <div dangerouslySetInnerHTML={{ __html: html }} />;  // EXPECT: js-inner-html
}

export function mount(el: HTMLElement, s: string) {
  el.innerHTML += s;  // EXPECT: js-inner-html
  if (el.innerHTML === s) return;
  el.textContent = s;
  const res = JSON.parse(s);
  return res;
}
'''))}},
    {"name": "model-and-trainer-lookalikes", "files": {"ml/trainer.py": (None, T('''class Trainer:
    def __init__(self, evaluator, sandbox):
        self.evaluator, self.sandbox = evaluator, sandbox

    def run(self, batch):
        self.sandbox.exec(batch)
        score = self.evaluator.eval(batch)
        text = f"eval={score:.3f} verify=False means strict"
        return score, text

    def exec_plan(self, plan):
        return [self.sandbox.execute(p) for p in plan]
'''))}},
    {"name": "debug-leftovers", "files": {"lib/dbg.py": (None, T('''def f(x):
    import pdb; pdb.set_trace()  # EXPECT: py-debug-leftover
    return x


def g(x):
    import ipdb; ipdb.set_trace()  # EXPECT: py-debug-leftover
    return x
'''))}},
    {"name": "refactor-moves-existing-risk", "files": {"app/old.py": (
        T('''def a():
    return 1


def risky(x):
    return eval(x)
'''),
        T('''def risky(x):
    return eval(x)


def a():
    return 1
'''))}},
]
