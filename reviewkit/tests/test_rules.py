from reviewkit.diff import parse_diff
from reviewkit.rules import (Finding, analyze_file, cyclomatic, is_test_path, python_code_view, python_line_code,
                             suppressed)
import ast


def fd_for(path, after, before=None):
    from reviewkit.bench import make_diff
    return parse_diff(make_diff({path: (before, after)}))[0]


def run(path, src, before=None):
    return analyze_file(fd_for(path, src, before), src)


def rules(path, src, before=None):
    return sorted((f.line, f.rule) for f in run(path, src, before)[0])


def test_code_view_blanks_strings_comments_and_docstrings():
    src = 'x = eval(s)  # eval(no)\ny = "eval(a)"\n"""\nexec(b)\n"""\nz = f"{eval(q)} text"\n'
    v = python_code_view(src)
    assert "eval(" in v[1] and "no" not in v[1]
    assert "eval" not in v[2]
    assert "exec" not in v[4]
    assert "eval(q)" in v[6]


def test_line_code_fallback_handles_escapes():
    assert python_line_code('a = "x\\"eval(" + eval(y)').count("eval(") == 1


def test_eval_not_flagged_for_method_or_string():
    assert rules("a.py", "model.eval()\ns = 'eval(1)'\n") == []
    assert rules("a.py", "r = eval(x)\n") == [(1, "py-eval")]


def test_torch_load_multiline_weights_only():
    ok = "import torch\nm = torch.load(\n    p,\n    weights_only=True,\n)\n"
    bad = "import torch\nm = torch.load(\n    p,\n    map_location='cpu',\n)\n"
    assert rules("a.py", ok) == []
    assert rules("a.py", bad) == [(2, "py-torch-load")]


def test_only_added_lines_are_reported():
    before = "def f(x):\n    return eval(x)\n"
    after = before + "\n\ndef g():\n    return 1\n"
    assert rules("a.py", after, before) == []


def test_secret_message_never_echoes_the_value():
    token = "AKIA" + "IOSFODNN7EXAMPLE"
    f = run("c.py", f'K = "{token}"\n')[0][0]
    assert f.rule == "secret-token" and token not in f.message and f.severity == "critical"


def test_placeholder_and_env_secrets_ignored():
    src = 'password = "changeme-changeme"\napi_key = os.environ["API_KEY"] or "abcdefgh12345"\nsecret = "xxxxxxxxxxxx"\n'
    assert rules("c.py", src) == []


def test_assignment_secrets_skipped_in_test_paths_but_tokens_not():
    assert rules("tests/test_x.py", 'password = "hunter2-hunter2"\n') == []
    assert is_test_path("pkg/test_x.py") and is_test_path("src/__tests__/a.js") and not is_test_path("src/latest.py")


def test_syntax_error_reported_on_added_line():
    f = run("a.py", "x = (\n")[0]
    assert [x.rule for x in f] == ["py-syntax-error"]


def test_working_tree_mismatch_skips_ast_with_note():
    fd = fd_for("a.py", "def f(a=[]):\n    pass\n")
    findings, notes = analyze_file(fd, "something else entirely\n")
    assert findings == [] and "does not match" in notes[0]


def test_ast_rules_need_full_source():
    fd = fd_for("a.py", "def f(a=[]):\n    pass\n")
    assert analyze_file(fd, None)[0] == []


def test_cyclomatic_counts_branches_boolops_and_ignores_nested_defs():
    fn = ast.parse("def f(a, b):\n    if a and b or a:\n        pass\n    def g():\n        if a:\n            pass\n    return [x for x in b if x]\n").body[0]
    assert cyclomatic(fn) == 1 + 1 + 2 + 2


def test_complexity_reported_on_first_added_line_of_function():
    body = "".join(f"    if a == {i}:\n        r += 1\n" for i in range(12))
    before = "def f(a):\n    r = 0\n" + body + "    return r\n"
    after = before.replace("    return r\n", "    r += 5\n    return r\n")
    assert rules("a.py", after, before) == [(2 + 24 + 1, "py-complexity")]


def test_mutable_default_in_lambda_and_kwonly():
    src = "f = lambda x, acc=[]: acc\n\n\ndef g(*, m={}):\n    return m\n"
    assert rules("a.py", src) == [(1, "py-mutable-default"), (4, "py-mutable-default")]


def test_requests_timeout_none_and_kwargs():
    src = "import requests\nrequests.get(u, timeout=None)\nrequests.get(u, **kw)\nrequests.get(u, timeout=3)\nrequests.get(u)\n"
    assert rules("a.py", src) == [(2, "py-requests-no-timeout"), (5, "py-requests-no-timeout")]


def test_pickle_name_imports_flag_calls():
    src = "from pickle import loads as ld\nx = ld(b)\n"
    assert rules("a.py", src) == [(2, "py-unsafe-deserialize")]


def test_is_literal_negative_numbers():
    assert rules("a.py", "if i is -1:\n    pass\nif i is None:\n    pass\n") == [(1, "py-is-literal")]


def test_sql_rules():
    src = ('cur.execute("select a from t where x=%s" % v)\n'
           'cur.execute("select a from t where x=%s", (v,))\n'
           'q = "SELECT * FROM t WHERE a = {}".format(a)\n'
           'm = f"update config set to {v}"\n'
           'u = f"UPDATE t SET a = {v}"\n')
    assert rules("a.py", src) == [(1, "py-sql-injection"), (3, "py-sql-injection"), (5, "py-sql-injection")]


def test_js_rules_and_comment_stripping():
    src = "el.innerHTML += s;\nel.innerHTML === s;\n// eval(x)\nvar a = eval(y);\n"
    assert rules("a.js", src) == [(1, "js-inner-html"), (4, "js-eval")]


def test_suppression_syntax():
    assert suppressed("x  # reviewkit: ignore", "py-eval")
    assert suppressed("x  # reviewkit: ignore[py-eval, other]", "py-eval")
    assert not suppressed("x  # reviewkit: ignore[other]", "py-eval")
    assert not suppressed("x", "py-eval")


def test_fingerprint_stable_across_line_shift_and_changes_with_text():
    a = Finding("p.py", 3, "r", "low", "m", line_text="  x = 1")
    b = Finding("p.py", 90, "r", "low", "m", line_text="x = 1")
    c = Finding("p.py", 3, "r", "low", "m", line_text="x = 2")
    assert a.fp == b.fp != c.fp


def test_binary_and_deleted_files_skipped():
    d = "diff --git a/g.py b/g.py\ndeleted file mode 100644\n--- a/g.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-eval(x)\n"
    assert analyze_file(parse_diff(d)[0], None)[0] == []


def test_assert_tuple_flagged_only_for_nonempty_tuple():
    assert rules("a.py", "assert (x > 1, 'msg')\n") == [(1, "py-assert-tuple")]
    assert rules("a.py", "assert x > 1, 'msg'\n") == []
    assert rules("a.py", "assert ()\n") == []


def test_return_in_finally():
    src = "def f():\n    try:\n        g()\n    finally:\n        return 1\n"
    assert rules("a.py", src) == [(5, "py-return-in-finally")]
    ok = "def f():\n    try:\n        g()\n    finally:\n        cleanup()\n    return 1\n"
    assert rules("a.py", ok) == []


def test_duplicate_dict_key():
    assert rules("a.py", "d = {\n    'a': 1,\n    'b': 2,\n    'a': 3,\n}\n") == [(4, "py-duplicate-dict-key")]
    assert rules("a.py", "d = {'a': 1, 'b': 2}\n") == []
    assert rules("a.py", "d = {1: 'x', '1': 'y'}\n") == []
    assert rules("a.py", "d = {**o, 'a': 1}\n") == []


def test_raise_notimplemented():
    assert rules("a.py", "def f():\n    raise NotImplemented\n") == [(2, "py-raise-notimplemented")]
    assert rules("a.py", "def f():\n    raise NotImplemented('x')\n") == [(2, "py-raise-notimplemented")]
    assert rules("a.py", "def f():\n    raise NotImplementedError\n") == []


def test_compare_none():
    assert rules("a.py", "if x == None:\n    pass\n") == [(1, "py-compare-none")]
    assert rules("a.py", "if x != None:\n    pass\n") == [(1, "py-compare-none")]
    assert rules("a.py", "if x is None:\n    pass\n") == []
    assert rules("a.py", "if x == 0:\n    pass\n") == []


def test_not_in_style():
    assert rules("a.py", "if not x in y:\n    pass\n") == [(1, "py-not-in-style")]
    assert rules("a.py", "if not x is None:\n    pass\n") == [(1, "py-not-in-style")]
    assert rules("a.py", "if x not in y:\n    pass\n") == []
    assert rules("a.py", "if not (x in y and z):\n    pass\n") == []


def test_self_compare():
    assert rules("a.py", "if x == x:\n    pass\n") == [(1, "py-self-compare")]
    assert rules("a.py", "if a < a:\n    pass\n") == [(1, "py-self-compare")]
    assert rules("a.py", "if x == y:\n    pass\n") == []
    assert rules("a.py", "if x.a == x.a:\n    pass\n") == []


def test_unreachable_code():
    assert rules("a.py", "def f():\n    return 1\n    print(2)\n") == [(3, "py-unreachable-code")]
    assert rules("a.py", "for i in x:\n    if i:\n        continue\n        y = 1\n") == [(4, "py-unreachable-code")]
    assert rules("a.py", "def f():\n    if a:\n        return 1\n    return 2\n") == []


def test_compare_bool():
    assert rules("a.py", "if x == True:\n    pass\n") == [(1, "py-compare-bool")]
    assert rules("a.py", "if x != False:\n    pass\n") == [(1, "py-compare-bool")]
    assert rules("a.py", "if x is True:\n    pass\n") == []
    assert rules("a.py", "if x == 1:\n    pass\n") == []


def test_duplicate_except():
    src = "try:\n    pass\nexcept ValueError:\n    raise\nexcept ValueError:\n    raise\n"
    assert rules("a.py", src) == [(5, "py-duplicate-except")]
    src = "try:\n    pass\nexcept (KeyError, ValueError):\n    raise\nexcept ValueError:\n    raise\n"
    assert rules("a.py", src) == [(5, "py-duplicate-except")]
    src = "try:\n    pass\nexcept KeyError:\n    raise\nexcept ValueError:\n    raise\n"
    assert rules("a.py", src) == []


def test_fstring_no_placeholder():
    assert rules("a.py", "x = f'hello'\n") == [(1, "py-fstring-no-placeholder")]
    assert rules("a.py", "x = f'hi {name}'\n") == []
    assert rules("a.py", "x = f'{v:>{w}}'\n") == []


def test_blocking_in_async():
    src = "import time\nasync def f():\n    time.sleep(1)\n"
    assert rules("a.py", src) == [(3, "py-blocking-in-async")]
    src = "import time\nasync def f():\n    def g():\n        time.sleep(1)\n    await h()\n"
    assert rules("a.py", src) == []
    assert rules("a.py", "import time\ndef f():\n    time.sleep(1)\n") == []


def test_useless_comparison():
    assert rules("a.py", "def f(x):\n    x == 3\n") == [(2, "py-useless-comparison")]
    assert rules("a.py", "def f(x):\n    assert x == 3\n    y = x == 3\n    return y\n") == []


def test_mktemp():
    assert rules("a.py", "import tempfile\np = tempfile.mktemp()\n") == [(2, "py-mktemp")]
    assert rules("a.py", "import tempfile\np = tempfile.mkstemp()\n") == []


def test_insecure_random_secret():
    bad = "import random\ntoken = ''.join(random.choice('abc') for _ in range(8))\n"
    assert rules("a.py", bad) == [(2, "py-insecure-random-secret")]
    assert rules("a.py", "import secrets\ntoken = secrets.token_urlsafe(16)\n") == []
    assert rules("a.py", "import random\ndelay = random.random()\n") == []


def test_ssl_unverified():
    assert rules("a.py", "import ssl\nc = ssl._create_unverified_context()\n") == [(2, "py-ssl-unverified")]
    assert rules("a.py", "import ssl\nc = ssl.create_default_context()\nc.check_hostname = False\n") == [(3, "py-ssl-unverified")]
    assert rules("a.py", "import ssl\nc = ssl.create_default_context()\n") == []


def test_tar_extractall():
    assert rules("a.py", "import tarfile\nt = tarfile.open(p)\nt.extractall(d)\n") == [(3, "py-tar-extractall")]
    assert rules("a.py", "import tarfile\nt = tarfile.open(p)\nt.extractall(d, filter='data')\n") == []


def test_chmod_world_writable():
    assert rules("a.py", "import os\nos.chmod(p, 0o777)\n") == [(2, "py-chmod-world-writable")]
    assert rules("a.py", "import os\nos.chmod(p, 0o600)\n") == []


def test_flask_debug():
    assert rules("a.py", "app.run(host='0.0.0.0', debug=True)\n") == [(1, "py-flask-debug")]
    assert rules("a.py", "app.run(debug=False)\n") == []


def test_xml_untrusted():
    assert rules("a.py", "import xml.etree.ElementTree as ET\nt = ET.fromstring(data)\n") == [(2, "py-xml-untrusted")]
    assert rules("a.py", "import defusedxml.ElementTree as D\nt = D.fromstring(data)\n") == []


def test_jinja_autoescape_off():
    assert rules("a.py", "env = Environment(loader=l, autoescape=False)\n") == [(1, "py-jinja-autoescape-off")]
    assert rules("a.py", "env = Environment(loader=l, autoescape=True)\n") == []


def test_jwt_no_verify():
    assert rules("a.py", "c = jwt.decode(tok, key, options={'verify_signature': False})\n") == [(1, "py-jwt-no-verify")]
    assert rules("a.py", "c = jwt.decode(tok, key, algorithms=['HS256'])\n") == []
