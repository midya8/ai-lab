import ast

from reviewkit.taint import taint_hits
from test_rules import rules


def hits(src):
    return [(l, r) for l, r, _ in taint_hits(ast.parse(src))]


def test_sql_via_variable_and_fstring():
    src = "def f(cur):\n    name = input()\n    q = f\"SELECT * FROM t WHERE n = '{name}'\"\n    cur.execute(q)\n"
    assert hits(src) == [(4, "py-taint-sql")]


def test_parameterized_query_is_clean():
    src = "def f(cur):\n    name = input()\n    cur.execute('SELECT * FROM t WHERE n = ?', (name,))\n"
    assert hits(src) == []


def test_flask_request_to_shell():
    src = "import os\ndef f():\n    host = request.args.get('h')\n    os.system('ping ' + host)\n"
    assert hits(src) == [(4, "py-taint-shell")]


def test_subprocess_needs_shell_true():
    base = "import subprocess\ndef f():\n    x = input()\n    subprocess.run(%s)\n"
    assert hits(base % "['ls', x]") == []
    assert hits(base % "'ls ' + x, shell=True") == [(4, "py-taint-shell")]


def test_sanitizer_and_reassignment_clear_taint():
    src = "def f(cur):\n    n = int(input())\n    cur.execute('SELECT ' + str(n))\n    s = input()\n    s = 'const'\n    eval(s)\n"
    assert hits(src) == []


def test_eval_of_argv_and_tuple_unpack():
    src = "import sys\ndef f():\n    a, b = sys.argv[1], 2\n    eval(a)\n"
    assert hits(src) == [(4, "py-taint-eval")]


def test_nested_function_scope_isolated():
    src = "def f():\n    x = input()\n    def g(x):\n        eval(x)\n"
    assert hits(src) == []


def test_integrated_with_added_lines_only():
    src = "def f(cur):\n    name = input()\n    q = 'SELECT a FROM t WHERE n=' + name\n    cur.execute(q)\n"
    assert (4, "py-taint-sql") in rules("a.py", src)
    # only the unrelated line 1 changes, so the pre-existing sink is not reported
    before = src.replace("def f(cur):", "def f(cur, z):")
    assert rules("a.py", src, before=before) == []
