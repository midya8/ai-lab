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
    LineRule("py-mktemp", "medium", ("python",), re.compile(r"\btempfile\.mktemp\s*\("),
             "`tempfile.mktemp` returns a name without creating the file, so another process can claim it first (race / symlink attack).",
             "Use `tempfile.mkstemp`, `NamedTemporaryFile` or `TemporaryDirectory`."),
    LineRule("py-insecure-random-secret", "medium", ("python",),
             re.compile(r"(?i)\b(?:token|secret|password|passwd|api_?key|session_?id|nonce|otp)\w*\s*=\s*.*\brandom\.(?:random|randint|choice|choices|randrange|getrandbits)\s*\("),
             "`random` is a predictable PRNG; values used as tokens, passwords or keys can be guessed.",
             "Use the `secrets` module (`secrets.token_urlsafe`, `secrets.choice`)."),
    LineRule("py-ssl-unverified", "high", ("python",),
             re.compile(r"\bssl\._create_unverified_context\b|\.verify_mode\s*=\s*ssl\.CERT_NONE\b|\bcheck_hostname\s*=\s*False\b"),
             "TLS certificate or hostname verification is disabled, enabling man-in-the-middle attacks.",
             "Use `ssl.create_default_context()` and keep verification on; supply a CA bundle for private CAs."),
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
    LineRule("py-tar-extractall", "high", ("python",),
             re.compile(r"\.extractall\s*\((?![^)]*\b(?:filter|members)\s*=)"),
             "`extractall` on an untrusted archive can write outside the target directory (path traversal, symlinks).",
             "Pass `filter=\"data\"` (Python 3.12+) or validate each member path before extracting."),
    LineRule("py-cleartext-protocol", "medium", ("python",),
             re.compile(r"\b(?:ftplib\.FTP|telnetlib\.Telnet|smtplib\.SMTP)\s*\("),
             "FTP, Telnet and plain SMTP send credentials and data unencrypted.",
             "Use `ftplib.FTP_TLS`, `smtplib.SMTP_SSL` (or `starttls()`), or SSH/SFTP instead."),
    LineRule("py-chmod-world-writable", "medium", ("python",),
             re.compile(r"\bos\.chmod\s*\([^)]*,\s*(?:0o?777|0o?666|stat\.S_IRWXO)\b"),
             "World-writable permissions let any local user modify the file.",
             "Use the narrowest mode (e.g. 0o600 for files, 0o750 for directories)."),
    LineRule("py-flask-debug", "high", ("python",), re.compile(r"\.run\s*\([^)]*\bdebug\s*=\s*True\b"),
             "Running a web app with debug=True exposes an interactive debugger that allows remote code execution.",
             "Drive debug mode from configuration and keep it off in production."),
    LineRule("py-xml-untrusted", "medium", ("python",), re.compile(r"\b(?:xml\.etree\.ElementTree|xml\.dom\.minidom|xml\.sax|ET|etree|minidom)\.(?:parse|fromstring|parseString|XML)\s*\("),
             "Stdlib XML parsers are vulnerable to entity-expansion (billion laughs) on untrusted input.",
             "Use `defusedxml` for untrusted XML."),
    LineRule("py-jinja-autoescape-off", "high", ("python",), re.compile(r"\bautoescape\s*=\s*False\b"),
             "Jinja2 with autoescape disabled renders user input as raw HTML (XSS).",
             "Use autoescape=True or select_autoescape()."),
    LineRule("py-jwt-no-verify", "high", ("python",), re.compile(r"\bjwt\.decode\s*\([^)]*\bFalse\b"),
             "Decoding a JWT with signature verification disabled accepts forged tokens.",
             "Verify the signature with a key and an explicit algorithms list."),
    LineRule("py-trust-remote-code", "high", ("python",), re.compile(r"\btrust_remote_code\s*=\s*True\b"),
             "trust_remote_code=True downloads and executes Python from the model repo at load time.",
             "Leave it False, or pin a reviewed revision= commit hash and audit the code first."),
    LineRule("py-dangerous-deserialization", "high", ("python",), re.compile(r"\ballow_dangerous_deserialization\s*=\s*True\b"),
             "allow_dangerous_deserialization=True unpickles the saved index; a tampered file runs arbitrary code.",
             "Only load indexes you wrote yourself from trusted storage, or rebuild the index from source data."),
    LineRule("py-bind-all-interfaces", "medium", ("python",),
             re.compile(r"""(?:\bhost\s*=\s*|\.bind\s*\(\s*\(\s*)["']0\.0\.0\.0["']"""),
             "Binding to 0.0.0.0 exposes the service on every network interface, including public ones.",
             "Bind to 127.0.0.1 by default and make the exposed host an explicit, reviewed configuration value.", view="raw", guard=_not_comment),
    LineRule("py-os-system", "medium", ("python",),
             re.compile(r"\bos\.(?:system|popen)\s*\((?!\s*[\"'][^\"'{}%]*[\"']\s*\))"),
             "os.system/os.popen run their argument through a shell; any interpolated value becomes command injection.",
             "Use `subprocess.run([\"cmd\", arg], check=True)` with an argument list.", view="raw", guard=_not_comment),
    LineRule("py-log-secret", "medium", ("python",),
             re.compile(r"(?i)\b(?:logger|logging|log)\.(?:debug|info|warning|error|critical)\s*\(.*\b(?:password|passwd|secret|api_?key|access_?token|auth_?token)\b"),
             "A log call mentions a credential-like name; logging it writes the secret to files and aggregators.",
             "Log an identifier or a redacted form (e.g. last 4 chars) instead of the secret value.", view="raw", guard=_not_comment),
    LineRule("py-hardcoded-tmp", "low", ("python",),
             re.compile(r"\bopen\s*\(\s*f?[\"']/tmp/"),
             "Opening a fixed path under /tmp is predictable: another user can pre-create it or symlink it (race/symlink attack).",
             "Use `tempfile.NamedTemporaryFile` / `tempfile.mkdtemp()` for a unique, private path.", view="raw", guard=_not_comment),
    LineRule("py-cleartext-http", "low", ("python",),
             re.compile(r"\b(?:requests|httpx|session)\.(?:get|post|put|delete|patch|request)\s*\(\s*f?[\"']http://(?!localhost\b|127\.0\.0\.1\b|0\.0\.0\.0\b|\[::1\])"),
             "Request to a plain http:// URL: traffic (and any credentials) can be read or modified in transit.",
             "Use https:// unless the host is local-only.", view="raw", guard=_not_comment),
    LineRule("py-hardcoded-bearer", "high", ("python",),
             re.compile(r"[\"']Bearer\s+[A-Za-z0-9._~+/=-]{20,}[\"']"),
             "A literal bearer token is committed in source; anyone with repo access can use it.",
             "Load the token from an environment variable or secret store and rotate this one.", view="raw", guard=_not_comment),
    LineRule("py-django-debug-true", "medium", ("python",),
             re.compile(r"^\s*DEBUG\s*=\s*True\b"),
             "`DEBUG = True` in settings exposes stack traces, settings and SQL to anyone who triggers an error.",
             "Read it from the environment (default False) and keep it off in production.", guard=_not_comment),
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

    nested_fstr: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Compare):
            add(node.lineno, "py-useless-comparison", "medium",
                "Comparison used as a statement has no effect (missing assert or assignment?).",
                "Use `assert ...`, assign the result, or delete the line.")
        if isinstance(node, ast.AsyncFunctionDef):
            stack = list(node.body)
            while stack:
                n = stack.pop()
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                    continue
                if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and isinstance(n.func.value, ast.Name)
                        and ((n.func.value.id == "time" and n.func.attr == "sleep")
                             or (n.func.value.id == "requests" and n.func.attr in _REQUEST_FUNCS))):
                    add(n.lineno, "py-blocking-in-async", "medium",
                        "Blocking call inside `async def` stalls the whole event loop.",
                        "Use `await asyncio.sleep(...)`, an async HTTP client, or `asyncio.to_thread(...)`.")
                stack.extend(ast.iter_child_nodes(n))
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for i, stmt in enumerate(block[:-1]):
                if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
                    add(block[i + 1].lineno, "py-unreachable-code", "medium",
                        f"Code after `{type(stmt).__name__.lower()}` on line {stmt.lineno} can never run.",
                        "Delete the dead statements or move the control-flow exit.")
                    break
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
            elif isinstance(node.type, ast.Name) and node.type.id == "BaseException" \
                    and not any(isinstance(n, ast.Raise) for st in node.body for n in ast.walk(st)):
                add(node.lineno, "py-catch-baseexception", "medium",
                    "`except BaseException` without re-raising swallows `KeyboardInterrupt`/`SystemExit`.",
                    "Catch `Exception` (or specific types), or re-raise after cleanup.")
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
            if isinstance(f, ast.Name) and f.id == "open" and node.args and not any(k.arg is None for k in node.keywords):
                mode = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for k in node.keywords:
                    if k.arg == "mode" and isinstance(k.value, ast.Constant):
                        mode = k.value.value
                if (len(node.args) < 2 or mode is not None) and (mode is None or isinstance(mode, str)) \
                        and "b" not in (mode or "") and len(node.args) < 4 \
                        and "encoding" not in {k.arg for k in node.keywords}:
                    add(node.lineno, "py-open-no-encoding", "low",
                        "`open()` in text mode without `encoding` depends on the platform locale (cp1252 on Windows).",
                        "Pass `encoding=\"utf-8\"` explicitly, or open in binary mode.")
            if (isinstance(f, ast.Attribute) and f.attr in ("load", "loadtxt", "genfromtxt")
                    and isinstance(f.value, ast.Name) and f.value.id in ("np", "numpy")
                    and any(k.arg == "allow_pickle" and isinstance(k.value, ast.Constant) and k.value.value is True
                            for k in node.keywords)):
                add(node.lineno, "py-numpy-allow-pickle", "high",
                    "`allow_pickle=True` makes `np.load` unpickle object arrays, which executes code from the file.",
                    "Store plain numeric arrays (the default `allow_pickle=False`), or only load files you produced.")
            if (isinstance(f, ast.Attribute) and f.attr == "run"
                    and isinstance(f.value, ast.Name) and f.value.id == "subprocess"
                    and not any(k.arg in ("check", None) for k in node.keywords)):
                add(node.lineno, "py-subprocess-no-check", "low",
                    "`subprocess.run` without `check=` silently ignores a non-zero exit status.",
                    "Pass `check=True`, or `check=False` explicitly and inspect `returncode`.")
            if (isinstance(f, ast.Attribute) and f.attr in ("read", "readlines", "readline", "write")
                    and isinstance(f.value, ast.Call) and isinstance(f.value.func, ast.Name)
                    and f.value.func.id == "open"):
                add(node.lineno, "py-open-unclosed", "low",
                    f"`open(...).{f.attr}()` never closes the file explicitly; the handle lingers until garbage collection.",
                    "Use `with open(...) as fh:` (or `Path.read_text()` / `write_text()`).")
            if (isinstance(f, ast.Attribute) and f.attr == "urlopen" and len(node.args) < 3
                    and not any(k.arg in ("timeout", None) for k in node.keywords)):
                add(node.lineno, "py-urlopen-no-timeout", "medium",
                    "`urlopen` without a timeout can hang forever on a stalled server.",
                    "Pass `timeout=` (for example 10 seconds).")
            if (isinstance(f, ast.Attribute) and f.attr in ("now", "today") and not node.args and not node.keywords
                    and ((isinstance(f.value, ast.Name) and f.value.id in ("datetime", "dt"))
                         or (isinstance(f.value, ast.Attribute) and f.value.attr == "datetime"))):
                add(node.lineno, "py-naive-now", "low",
                    f"`datetime.{f.attr}()` without a timezone returns a naive local time that breaks across servers and DST.",
                    "Use `datetime.now(timezone.utc)` and convert to local time only for display.")
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "requests"
                    and f.attr in _REQUEST_FUNCS):
                kws = {k.arg for k in node.keywords}
                none_timeout = any(k.arg == "timeout" and isinstance(k.value, ast.Constant) and k.value.value is None
                                   for k in node.keywords)
                if none_timeout or ("timeout" not in kws and None not in kws):
                    add(node.lineno, "py-requests-no-timeout", "medium",
                        "`requests` has no default timeout; a stalled server hangs this call forever.",
                        "Pass `timeout=(connect, read)`, e.g. `timeout=10`.")
        elif isinstance(node, ast.Attribute) and node.attr in ("utcnow", "utcfromtimestamp") and isinstance(
                node.value, (ast.Name, ast.Attribute)) and (
                (isinstance(node.value, ast.Name) and node.value.id in ("datetime", "dt"))
                or (isinstance(node.value, ast.Attribute) and node.value.attr == "datetime")):
            add(node.lineno, "py-utcnow", "low",
                f"`datetime.{node.attr}` returns a naive datetime and is deprecated since Python 3.12.",
                "Use `datetime.now(timezone.utc)` / `datetime.fromtimestamp(ts, timezone.utc)`.")
        elif isinstance(node, ast.Compare) and any(isinstance(o, (ast.Eq, ast.NotEq)) for o in node.ops) and any(
                isinstance(x, ast.Constant) and isinstance(x.value, float) and not float(x.value).is_integer()
                for x in [node.left, *node.comparators]):
            add(node.lineno, "py-float-equality", "low",
                "Exact `==` / `!=` against a non-integer float literal is fragile because of rounding error.",
                "Use `math.isclose(a, b, rel_tol=...)` or compare within a tolerance.")
        elif isinstance(node, ast.Assert):
            if isinstance(node.test, ast.Tuple) and node.test.elts:
                add(node.lineno, "py-assert-tuple", "high",
                    "`assert (cond, msg)` asserts a non-empty tuple, which is always true; the check never fails.",
                    "Write `assert cond, msg` without the parentheses around both.")
        elif isinstance(node, ast.Raise):
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id == "NotImplemented":
                add(node.lineno, "py-raise-notimplemented", "high",
                    "`NotImplemented` is a comparison sentinel, not an exception; raising it is a TypeError at runtime.",
                    "Raise `NotImplementedError` instead.")
        elif isinstance(node, ast.JoinedStr):
            if id(node) not in nested_fstr:
                for v in node.values:
                    if isinstance(v, ast.FormattedValue) and isinstance(v.format_spec, ast.JoinedStr):
                        nested_fstr.add(id(v.format_spec))
                if not any(isinstance(v, ast.FormattedValue) for v in node.values):
                    add(node.lineno, "py-fstring-no-placeholder", "low",
                        "f-string has no placeholders; the `f` prefix does nothing and often hides a forgotten `{...}`.",
                        "Drop the `f` prefix or add the intended placeholder.")
        elif isinstance(node, ast.Try):
            seen_exc = set()
            for h in node.handlers:
                names = [h.type] if h.type is not None and not isinstance(h.type, ast.Tuple) else (list(h.type.elts) if h.type is not None else [])
                for t in names:
                    nm = ast.unparse(t)
                    if nm in seen_exc:
                        add(h.lineno, "py-duplicate-except", "medium",
                            f"`except {nm}` already handled by an earlier clause in this try; this handler never runs.",
                            "Remove the duplicate or catch the intended exception type.")
                    seen_exc.add(nm)
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
                if isinstance(op, (ast.Eq, ast.NotEq)) and isinstance(comp, ast.Constant) and isinstance(comp.value, bool):
                    add(node.lineno, "py-compare-bool", "low",
                        f"Comparing with `{comp.value}` using `==` is redundant and misfires for 1/0 and numpy values.",
                        "Test truthiness directly (`if x:` / `if not x:`), or use `is True` / `is False` if the exact bool matters.")
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
    "py-catch-baseexception": ("medium", "`except BaseException` that never re-raises."),
    "py-swallowed-exception": ("medium", "Exception caught and discarded with `pass`."),
    "py-requests-no-timeout": ("medium", "`requests.*` call without `timeout`."),
    "py-is-literal": ("medium", "`is` / `is not` against a str/int/float/bytes literal."),
    "py-raise-notimplemented": ("high", "`raise NotImplemented` instead of `NotImplementedError`."),
    "py-assert-tuple": ("high", "`assert` on a non-empty tuple is always true."),
    "py-float-equality": ("low", "`==` / `!=` against a non-integer float literal."),
    "py-open-no-encoding": ("low", "Text-mode `open()` without `encoding=`."),
    "py-urlopen-no-timeout": ("medium", "`urllib.request.urlopen` without a timeout."),
    "py-numpy-allow-pickle": ("high", "`np.load(..., allow_pickle=True)` unpickles untrusted data."),
    "py-naive-now": ("low", "`datetime.now()` / `today()` with no timezone."),
    "py-utcnow": ("low", "`datetime.utcnow()` / `utcfromtimestamp` (naive, deprecated)."),
    "py-return-in-finally": ("medium", "`return` inside `finally` swallows exceptions."),
    "py-compare-none": ("low", "`== None` / `!= None` instead of `is None`."),
    "py-raise-notimplemented": ("medium", "`raise NotImplemented` (should be NotImplementedError)."),
    "py-not-in-style": ("low", "`not a in b` / `not a is b` instead of `not in` / `is not`."),
    "py-self-compare": ("low", "Name compared with itself."),
    "py-subprocess-no-check": ("low", "`subprocess.run` without `check=`."),
    "py-open-unclosed": ("low", "`open(...).read()/write()` without a with-block."),
    "py-duplicate-except": ("medium", "Exception type caught twice in one try."),
    "py-blocking-in-async": ("medium", "time.sleep / requests call inside async def."),
    "py-useless-comparison": ("medium", "Comparison as a bare statement has no effect."),
    "py-insecure-random-secret": ("medium", "`random` used to generate a token/secret/password."),
    "py-tar-extractall": ("high", "Archive extractall without filter/members (path traversal)."),
    "py-cleartext-protocol": ("medium", "ftplib.FTP / telnetlib.Telnet / smtplib.SMTP (unencrypted)."),
    "py-chmod-world-writable": ("medium", "os.chmod with 0o777/0o666."),
    "py-flask-debug": ("high", "App run with debug=True."),
    "py-jinja-autoescape-off": ("high", "Jinja2 autoescape=False."),
    "py-jwt-no-verify": ("high", "jwt.decode with signature verification disabled."),
    "py-dangerous-deserialization": ("high", "allow_dangerous_deserialization=True (pickle-backed index load)."),
    "py-trust-remote-code": ("high", "Hugging Face trust_remote_code=True."),
    "py-os-system": ("medium", "os.system/os.popen with a non-constant command."),
    "py-hardcoded-bearer": ("high", "Literal Authorization: Bearer token in source."),
    "py-django-debug-true": ("medium", "Settings-style DEBUG = True literal."),
    "py-cleartext-http": ("low", "HTTP client call to a non-local http:// URL."),
    "py-hardcoded-tmp": ("low", "open() of a fixed /tmp path."),
    "py-log-secret": ("medium", "Log call that includes a password/secret/token variable."),
    "py-bind-all-interfaces": ("medium", "Server bound to 0.0.0.0."),
    "py-xml-untrusted": ("medium", "Stdlib XML parser on possibly untrusted input."),
    "py-ssl-unverified": ("high", "TLS certificate/hostname verification disabled via ssl."),
    "py-fstring-no-placeholder": ("low", "f-string without any placeholder."),
    "py-compare-bool": ("low", "`== True` / `== False` instead of truthiness."),
    "py-unreachable-code": ("medium", "Statements after return/raise/continue/break."),
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
