"""Deterministic review rules.

Line rules run on a *code view* of each added line (strings and comments blanked
out, via the tokenizer when the full file is available) so `# never call eval(x)`
and `"eval("` do not trigger. AST rules need the full new file. Every rule only
reports on lines the PR actually added.
"""
from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .diff import FileDiff

SEVERITIES = ["info", "low", "medium", "high", "critical"]


def sev_rank(s: str) -> int:
    return SEVERITIES.index(s)


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    rule: str
    severity: str
    message: str
    suggestion: Optional[str] = None
    source: str = "rule"  # "rule" or "llm"
    line_text: str = ""

    @property
    def fp(self) -> str:
        """Stable fingerprint: survives line-number shifts, changes when the line does."""
        basis = f"{self.path}|{self.rule}|{self.line_text.strip()}"
        if self.source == "llm":
            basis += "|" + self.message
        return hashlib.sha1(basis.encode()).hexdigest()[:12]


LANG_BY_EXT = {
    ".py": "python",
    ".js": "js", ".jsx": "js", ".ts": "js", ".tsx": "js", ".mjs": "js", ".cjs": "js",
}


def language_of(path: str) -> Optional[str]:
    for ext, lang in LANG_BY_EXT.items():
        if path.endswith(ext):
            return lang
    return None


def is_test_path(path: str) -> bool:
    p = path.lower()
    return bool(re.search(r"(^|/)(tests?|fixtures?|__tests__|spec)/|(^|/)test_[^/]*$|_test\.[a-z]+$|\.test\.[a-z]+$", p))


# --------------------------------------------------------------------------- code views

def _blank_range(grid: List[List[str]], s: Tuple[int, int], e: Tuple[int, int]) -> None:
    (sr, sc), (er, ec) = s, e
    for r in range(sr, er + 1):
        row = grid[r - 1]
        lo = sc if r == sr else 0
        hi = ec if r == er else len(row)
        for c in range(lo, min(hi, len(row))):
            row[c] = " "


def python_code_view(source: str) -> Optional[Dict[int, str]]:
    """Line number -> line with string contents and comments blanked. None if untokenizable."""
    grid = [list(l) for l in source.split("\n")]
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT) or (fstring_middle and tok.type == fstring_middle):
                _blank_range(grid, tok.start, tok.end)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    return {i + 1: "".join(r) for i, r in enumerate(grid)}


def python_line_code(line: str) -> str:
    """Single-line approximation used when the full file is unavailable."""
    out, quote, i = [], None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            if ch == quote:
                quote = None
                out.append(ch)
            else:
                out.append(" ")
        elif ch == "#":
            break
        elif ch in "\"'":
            quote = ch
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def js_line_code(line: str) -> str:
    out, quote, i = [], None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                out.append("  ")
                i += 2
                continue
            out.append(ch if ch == quote else " ")
            if ch == quote:
                quote = None
        elif line.startswith("//", i):
            break
        elif ch in "\"'`":
            quote = ch
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


@dataclass
class FileContext:
    path: str
    lang: Optional[str]
    added: Dict[int, str]
    source_lines: Optional[List[str]] = None
    code: Dict[int, str] = field(default_factory=dict)

    def code_at(self, n: int) -> str:
        if n in self.code:
            return self.code[n]
        text = self.added.get(n, "")
        return python_line_code(text) if self.lang == "python" else js_line_code(text)

    def call_text(self, n: int, max_lines: int = 8) -> str:
        """Code of line n plus continuation lines until its brackets balance."""
        if self.source_lines is None:
            return self.code_at(n)
        parts, depth = [], 0
        for k in range(n, min(n + max_lines, len(self.source_lines)) + 1):
            c = self.code_at(k) if k in self.code else self.source_lines[k - 1]
            parts.append(c)
            depth += sum(c.count(x) for x in "([{") - sum(c.count(x) for x in ")]}")
            if depth <= 0:
                break
        return " ".join(parts)


# --------------------------------------------------------------------------- line rules

@dataclass
class LineRule:
    id: str
    severity: str
    langs: Tuple[str, ...]
    pattern: "re.Pattern[str]"
    message: str
    suggestion: Optional[str] = None
    view: str = "code"  # "code" or "raw"
    guard: Optional[Callable[[FileContext, int], bool]] = None


def _no_weights_only(ctx: FileContext, n: int) -> bool:
    return "weights_only" not in ctx.call_text(n)


def _no_safe_loader(ctx: FileContext, n: int) -> bool:
    return "SafeLoader" not in ctx.call_text(n) and "CSafeLoader" not in ctx.call_text(n)


def _no_usedforsecurity(ctx: FileContext, n: int) -> bool:
    return "usedforsecurity" not in ctx.call_text(n)


def _not_def(ctx: FileContext, n: int) -> bool:
    return not re.search(r"\bdef\s+(?:eval|exec)\b", ctx.code_at(n))


def _not_comment(ctx: FileContext, n: int) -> bool:
    t = ctx.added.get(n, "").lstrip()
    return not t.startswith(("#", "//", "*"))


_SQL = r"(?:select\s.+\sfrom\s+\w+|insert\s+into\s+\w+|update\s+\w+\s+set\s+\w+\s*=|delete\s+from\s+\w+)"

LINE_RULES: List[LineRule] = [
    LineRule("py-eval", "high", ("python",), re.compile(r"(?<![\w.])(?:eval|exec)\s*\("),
             "`eval`/`exec` runs arbitrary code; if the input can be influenced by a user this is remote code execution.",
             "Use `ast.literal_eval` for literals, or a dispatch table / explicit parser.", guard=_not_def),
    LineRule("py-shell-true", "high", ("python",), re.compile(r"\bshell\s*=\s*True\b"),
             "`shell=True` passes the command through a shell; interpolated input becomes command injection.",
             "Pass an argument list (`subprocess.run([\"cmd\", arg])`) and drop `shell=True`."),
    LineRule("py-unsafe-deserialize", "high", ("python",), re.compile(r"(?<![\w.])(?:pickle|cPickle|dill|marshal|shelve)\.(?:loads?|Unpickler)\b"),
             "Deserialising with pickle-style formats executes code embedded in the data.",
             "Use JSON (or a schema-validated format) for anything that crosses a trust boundary."),
    LineRule("py-torch-load", "medium", ("python",), re.compile(r"(?<![\w.])torch\.load\s*\("),
             "`torch.load` unpickles arbitrary objects unless `weights_only=True`; untrusted checkpoints can run code.",
             "Pass `weights_only=True` (or load safetensors).", guard=_no_weights_only),
    LineRule("py-yaml-load", "high", ("python",), re.compile(r"(?<![\w.])yaml\.load\s*\("),
             "`yaml.load` without a safe loader can construct arbitrary Python objects.",
             "Use `yaml.safe_load(...)`.", guard=_no_safe_loader),
    LineRule("py-tls-verify-off", "high", ("python",), re.compile(r"\bverify\s*=\s*False\b"),
             "TLS certificate verification is disabled, which allows man-in-the-middle attacks.",
             "Remove `verify=False`; point `verify=` at a CA bundle if you use a private CA."),
    LineRule("py-weak-hash", "low", ("python",), re.compile(r"\bhashlib\.(?:md5|sha1)\s*\("),
             "MD5/SHA-1 are broken for security purposes (passwords, signatures, integrity checks).",
             "Use SHA-256+ (or a password KDF such as scrypt/argon2); pass `usedforsecurity=False` if it is only a cache key.",
             guard=_no_usedforsecurity),
    LineRule("py-debug-leftover", "medium", ("python",), re.compile(r"(?<![\w.])breakpoint\s*\(\)|\bpdb\.set_trace\s*\(|^\s*import\s+pdb\b|\bipdb\.set_trace\s*\("),
             "Debugger call left in the change; it will hang a non-interactive process.", "Remove it."),
    LineRule("py-sql-injection", "high", ("python",),
             re.compile(r"\.execute(?:many|script)?\(\s*(?:f[\"']|[\"'][^\"']*[\"']\s*(?:%|\+|\.format\s*\())"
                        r"|(?i:\bf[\"'].*\b" + _SQL + r".*\{)"
                        r"|(?i:[\"'][^\"']*\b" + _SQL + r"[^\"']*[\"']\s*(?:\.format\s*\(|%\s*[\w(]))"),
             "SQL is built by string formatting; user-controlled values become SQL injection.",
             "Use parameterised queries: `cur.execute(\"... WHERE id = %s\", (value,))`.", view="raw", guard=_not_comment),
    LineRule("js-eval", "high", ("js",), re.compile(r"(?<![\w.])eval\s*\(|\bnew\s+Function\s*\("),
             "`eval`/`new Function` executes arbitrary code.", "Parse the data (JSON.parse) or use a lookup table."),
    LineRule("js-inner-html", "medium", ("js",), re.compile(r"\.innerHTML\s*\+?=(?!=)|dangerouslySetInnerHTML"),
             "Assigning HTML from strings is an XSS risk when any part is user-controlled.",
             "Use `textContent`, or sanitise the HTML (e.g. DOMPurify) first."),
    LineRule("merge-conflict", "high", ("python", "js"), re.compile(r"^(?:<{7}|>{7})(?: |$)|^={7}(?:\s|$)"),
             "Merge-conflict marker committed.", "Resolve the conflict and remove the marker.", view="raw"),
]

# --------------------------------------------------------------------------- secrets

_SPECIFIC_SECRETS = [
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("API key (sk- prefix)", re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
]
_ASSIGNED_SECRET = re.compile(
    r"(?i)\b(?:[a-z0-9_]*(?:password|passwd|secret|api[_-]?key|access[_-]?key|auth[_-]?token|token))\b[\"']?\s*[:=]\s*[\"']([^\"'\s]{8,})[\"']")
_PLACEHOLDER = re.compile(r"(?i)changeme|change_me|placeholder|example|your[_-]|<.*>|\$\{|\{\{|xxxx|dummy|redacted|fake|todo|\*{4,}|^env:|\.\.\.")


def _looks_like_secret_value(v: str) -> bool:
    if _PLACEHOLDER.search(v):
        return False
    if len(set(v)) < 5:
        return False
    return True


def secret_findings(ctx: FileContext) -> List[Finding]:
    out: List[Finding] = []
    for n, text in sorted(ctx.added.items()):
        hit = False
        for label, rx in _SPECIFIC_SECRETS:
            if rx.search(text):
                out.append(Finding(ctx.path, n, "secret-token", "critical",
                                   f"Looks like a committed {label}. The value is not repeated here; rotate it, since it is now in git history.",
                                   "Load it from an environment variable or a secret manager.", line_text=text))
                hit = True
                break
        if hit or is_test_path(ctx.path):
            continue
        m = _ASSIGNED_SECRET.search(text)
        if m and _looks_like_secret_value(m.group(1)) and "environ" not in text and "getenv" not in text:
            out.append(Finding(ctx.path, n, "secret-assignment", "high",
                               "A credential-looking name is assigned a string literal. If this is real, rotate it and load it from the environment.",
                               "Read it with `os.environ[\"NAME\"]` / a secret manager.", line_text=text))
    return out


# --------------------------------------------------------------------------- AST rules

_MUTABLE_CALLS = {"list", "dict", "set", "defaultdict", "OrderedDict", "deque", "Counter"}
_REQUEST_FUNCS = {"get", "post", "put", "delete", "patch", "head", "request", "options"}


def _is_mutable_default(d: ast.AST) -> bool:
    return isinstance(d, (ast.List, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp)) or (
        isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id in _MUTABLE_CALLS)


_PICKLE_MODULES = {"pickle", "cPickle", "dill", "marshal", "shelve"}
_PICKLE_NAMES = {"load", "loads", "Unpickler"}


def cyclomatic(fn: ast.AST) -> int:
    score = 1

    class V(ast.NodeVisitor):
        def generic_visit(self, node):
            nonlocal score
            if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.IfExp, ast.Assert)):
                score += 1
            elif isinstance(node, ast.BoolOp):
                score += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                score += 1 + len(node.ifs)
            elif isinstance(node, ast.match_case):
                score += 1
            super().generic_visit(node)

        def visit_FunctionDef(self, node):
            if node is fn:
                self.generic_visit(node)  # nested functions are scored on their own

        visit_AsyncFunctionDef = visit_FunctionDef
        visit_Lambda = lambda self, node: None  # noqa: E731

    V().visit(fn)
    return score


def _first_added(added: Set[int], lo: int, hi: int) -> Optional[int]:
    hits = [n for n in added if lo <= n <= hi]
    return min(hits) if hits else None


def ast_findings(ctx: FileContext, source: str, complexity_threshold: int = 10) -> List[Finding]:
    out: List[Finding] = []
    added = set(ctx.added)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        ln = e.lineno or 0
        if ln in added:
            out.append(Finding(ctx.path, ln, "py-syntax-error", "critical", f"File does not parse: {e.msg}.", line_text=ctx.added[ln]))
        return out

    def add(line: int, rule: str, sev: str, msg: str, sug: Optional[str] = None) -> None:
        if line in added:
            out.append(Finding(ctx.path, line, rule, sev, msg, sug, line_text=ctx.added[line]))

    pickle_aliases: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in _PICKLE_MODULES:
            for a in node.names:
                if a.name in _PICKLE_NAMES:
                    pickle_aliases.add(a.asname or a.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            for d in list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]:
                if _is_mutable_default(d):
                    add(d.lineno, "py-mutable-default", "medium",
                        "Mutable default argument in a lambda is created once and shared across calls.",
                        "Use `None` and build the object in a regular function.")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [d for d in node.args.kw_defaults if d is not None]
            for d in defaults:
                if _is_mutable_default(d):
                    add(d.lineno, "py-mutable-default", "medium",
                        f"Mutable default argument in `{node.name}` is created once and shared across calls.",
                        "Default to `None` and create the object inside the function.")
            score = cyclomatic(node)
            if score > complexity_threshold:
                span_end = getattr(node, "end_lineno", node.lineno)
                first = _first_added(added, node.lineno, span_end)
                if first is not None:
                    add(first, "py-complexity", "low",
                        f"`{node.name}` has cyclomatic complexity {score} (threshold {complexity_threshold}); it is hard to test and review.",
                        "Split it into smaller functions.")
        elif isinstance(node, ast.ExceptHandler):
            if node.type is None:
                add(node.lineno, "py-bare-except", "medium",
                    "Bare `except:` also catches `KeyboardInterrupt`/`SystemExit` and hides bugs.",
                    "Catch the specific exceptions you expect.")
            body_trivial = all(isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis)
                               for s in node.body)
            if body_trivial:
                line = node.lineno if node.lineno in added else (node.body[0].lineno if node.body[0].lineno in added else None)
                if line is not None:
                    add(line, "py-swallowed-exception", "medium",
                        "The exception is caught and silently discarded; failures will be invisible.",
                        "At least log it (`logger.exception(...)`), or let it propagate.")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in pickle_aliases:
                add(node.lineno, "py-unsafe-deserialize", "high",
                    "Deserialising with pickle-style formats executes code embedded in the data.",
                    "Use JSON (or a schema-validated format) for anything that crosses a trust boundary.")
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "requests"
                    and f.attr in _REQUEST_FUNCS):
                kws = {k.arg for k in node.keywords}
                none_timeout = any(k.arg == "timeout" and isinstance(k.value, ast.Constant) and k.value.value is None
                                   for k in node.keywords)
                if none_timeout or ("timeout" not in kws and None not in kws):
                    add(node.lineno, "py-requests-no-timeout", "medium",
                        "`requests` has no default timeout; a stalled server hangs this call forever.",
                        "Pass `timeout=(connect, read)`, e.g. `timeout=10`.")
        elif isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Tuple) and node.test.elts:
                add(node.lineno, "py-assert-tuple", "high",
                    "`assert (cond, msg)` asserts a non-empty tuple, which is always true; the check never fails.",
                    "Write `assert cond, msg` without the parentheses around both.")
        elif isinstance(node, ast.Try):
            for stmt in node.finalbody:
                for sub in ast.walk(stmt):
                    if isinstance(sub, ast.Return):
                        add(sub.lineno, "py-return-in-finally", "medium",
                            "`return` inside `finally` silently swallows any in-flight exception.",
                            "Move the return out of the `finally` block.")
        elif isinstance(node, ast.Raise):
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplemented":
                add(node.lineno, "py-raise-notimplemented", "medium",
                    "`raise NotImplemented` raises a TypeError; NotImplemented is not an exception.",
                    "Use `raise NotImplementedError(...)`.")
        elif isinstance(node, ast.Dict):
            seen: Dict[Any, ast.AST] = {}
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, (str, bytes, int, float)) and not isinstance(k.value, bool):
                    key = (type(k.value).__name__, k.value)
                    if key in seen:
                        add(k.lineno, "py-duplicate-dict-key", "medium",
                            f"Dict literal repeats the key {k.value!r}; the earlier value is silently discarded.",
                            "Remove the duplicate or fix the intended key.")
                    seen[key] = v
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Compare) \
                    and len(node.operand.ops) == 1 and isinstance(node.operand.ops[0], (ast.In, ast.Is)):
                word = "not in" if isinstance(node.operand.ops[0], ast.In) else "is not"
                add(node.lineno, "py-not-in-style", "low",
                    f"`not a {'in' if word == 'not in' else 'is'} b` reads ambiguously; use the dedicated operator.",
                    f"Write `a {word} b`.")
        elif isinstance(node, ast.Compare):
            if len(node.ops) == 1 and isinstance(node.left, ast.Name) and isinstance(node.comparators[0], ast.Name) \
                    and node.left.id == node.comparators[0].id \
                    and isinstance(node.ops[0], (ast.Eq, ast.NotEq, ast.Lt, ast.Gt, ast.LtE, ast.GtE, ast.Is, ast.IsNot)):
                add(node.lineno, "py-self-compare", "low",
                    f"`{node.left.id}` is compared with itself; the result is constant (except NaN checks).",
                    "Fix the typo'd operand, or use `math.isnan` for NaN checks.")
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(comp, ast.UnaryOp) and isinstance(comp.op, (ast.USub, ast.UAdd)):
                    comp = comp.operand
                if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(comp, ast.Constant) and \
                        isinstance(comp.value, (str, bytes, int, float)) and not isinstance(comp.value, bool):
                    add(node.lineno, "py-is-literal", "medium",
                        "`is` compares identity, not value; comparing with a literal is unreliable.",
                        "Use `==` / `!=`.")
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) and comp.value is None:
                    add(node.lineno, "py-compare-none", "low",
                        "`== None` / `!= None` can be overridden by `__eq__` (numpy/pandas return arrays).",
                        "Use `is None` / `is not None`.")
    from .taint import taint_hits
    for line, rule, msg in taint_hits(tree):
        add(line, rule, "high", msg, "Pass values as bound parameters / argument lists, or validate against an allow-list.")
    return out


RULE_TABLE: Dict[str, Tuple[str, str]] = {r.id: (r.severity, r.message) for r in LINE_RULES}
RULE_TABLE.update({
    "secret-token": ("critical", "Committed token of a recognisable format (AWS, GitHub, private key, sk-, Slack)."),
    "secret-assignment": ("high", "Credential-looking name assigned a string literal (skipped in test paths)."),
    "py-mutable-default": ("medium", "Mutable default argument."),
    "py-complexity": ("low", "Function over the cyclomatic-complexity threshold."),
    "py-bare-except": ("medium", "Bare `except:`."),
    "py-swallowed-exception": ("medium", "Exception caught and discarded with `pass`."),
    "py-requests-no-timeout": ("medium", "`requests.*` call without `timeout`."),
    "py-is-literal": ("medium", "`is` / `is not` against a str/int/float/bytes literal."),
    "py-assert-tuple": ("high", "`assert` on a non-empty tuple is always true."),
    "py-return-in-finally": ("medium", "`return` inside `finally` swallows exceptions."),
    "py-compare-none": ("low", "`== None` / `!= None` instead of `is None`."),
    "py-raise-notimplemented": ("medium", "`raise NotImplemented` (should be NotImplementedError)."),
    "py-not-in-style": ("low", "`not a in b` / `not a is b` instead of `not in` / `is not`."),
    "py-self-compare": ("low", "Name compared with itself."),
    "py-duplicate-dict-key": ("medium", "Dict literal repeats a constant key."),
    "py-syntax-error": ("critical", "Changed Python file does not parse."),
    "py-taint-sql": ("high", "Untrusted input flows into SQL query text."),
    "py-taint-shell": ("high", "Untrusted input flows into a shell command."),
    "py-taint-eval": ("high", "Untrusted input flows into eval/exec."),
})


# --------------------------------------------------------------------------- driver

_IGNORE_RE = re.compile(r"reviewkit:\s*ignore(?:\[([^\]]*)\])?")


def suppressed(line_text: str, rule: str) -> bool:
    m = _IGNORE_RE.search(line_text)
    if not m:
        return False
    if m.group(1) is None:
        return True
    return rule in {r.strip() for r in m.group(1).split(",")}


def analyze_file(fd: FileDiff, source: Optional[str], complexity_threshold: int = 10) -> Tuple[List[Finding], List[str]]:
    """Run every rule over one changed file. Returns (findings, notes)."""
    notes: List[str] = []
    lang = language_of(fd.path)
    added = fd.added()
    if fd.binary or fd.status == "deleted" or not added:
        return [], notes
    ctx = FileContext(fd.path, lang, added)
    if source is not None and lang == "python":
        src_lines = source.split("\n")
        if all(n <= len(src_lines) and src_lines[n - 1] == t for n, t in added.items()):
            ctx.source_lines = src_lines
            view = python_code_view(source)
            if view is not None:
                ctx.code = view
        else:
            notes.append(f"{fd.path}: working tree does not match the diff, so AST rules were skipped (check out the PR head).")
            source = None
    findings: List[Finding] = []
    for n in sorted(added):
        raw = added[n]
        for rule in LINE_RULES:
            if lang is None or lang not in rule.langs:
                continue
            target = raw if rule.view == "raw" else ctx.code_at(n)
            if rule.pattern.search(target) and (rule.guard is None or rule.guard(ctx, n)):
                findings.append(Finding(fd.path, n, rule.id, rule.severity, rule.message, rule.suggestion, line_text=raw))
    findings.extend(secret_findings(ctx))
    if lang == "python" and source is not None and ctx.source_lines is not None:
        findings.extend(ast_findings(ctx, source, complexity_threshold))
    return findings, notes
