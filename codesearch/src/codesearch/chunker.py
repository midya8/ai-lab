"""Parse Python source into searchable chunks: one per function, method, class and module doc."""
from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import dataclass, field


@dataclass
class Chunk:
    path: str
    qualname: str
    kind: str  # function | method | class | module
    start: int
    end: int
    signature: str = ""
    doc: str = ""
    ids: list[str] = field(default_factory=list)  # identifiers used in the body, with repeats
    strings: list[str] = field(default_factory=list)  # short string literals
    comments: list[str] = field(default_factory=list)
    is_test: bool = False

    @property
    def id(self) -> str:
        return f"{self.path}::{self.qualname}"

    @property
    def name(self) -> str:
        return self.qualname.split(".")[-1]

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    base = parts[-1]
    return (
        any(p in ("tests", "test") for p in parts[:-1])
        or base.startswith("test_")
        or base.endswith("_test.py")
        or base == "conftest.py"
    )


def _comments_by_line(source: str) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT:
                text = tok.string.lstrip("#").strip()
                if text and not text.startswith(("!", "type:", "noqa", "pragma", "reviewkit:")):
                    out[tok.start[0]] = text
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return out


def _signature(node: ast.AST) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){ret}"


def _decorators(node: ast.AST) -> list[str]:
    return [ast.unparse(d) for d in getattr(node, "decorator_list", [])]


def _walk_body(stmts: list[ast.stmt], skip_first_doc: bool) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    strings: list[str] = []
    for i, stmt in enumerate(stmts):
        if i == 0 and skip_first_doc and _is_doc_stmt(stmt):
            continue
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name):
                ids.append(n.id)
            elif isinstance(n, ast.Attribute):
                ids.append(n.attr)
            elif isinstance(n, ast.arg):
                ids.append(n.arg)
            elif isinstance(n, ast.keyword) and n.arg:
                ids.append(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ids.append(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    ids.append(a.name.split(".")[-1])
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and 2 < len(n.value) <= 100:
                strings.append(n.value)
    return ids, strings


def _is_doc_stmt(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _end(node: ast.AST) -> int:
    return getattr(node, "end_lineno", None) or node.lineno


def chunk_source(source: str, path: str) -> list[Chunk]:
    """Chunk one file. Raises SyntaxError if the file does not parse."""
    tree = ast.parse(source)
    comments = _comments_by_line(source)
    test = is_test_path(path)
    chunks: list[Chunk] = []

    def comment_text(lo: int, hi: int) -> list[str]:
        return [t for ln, t in comments.items() if lo <= ln <= hi]

    mod_doc = ast.get_docstring(tree)
    if mod_doc:
        top = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        chunks.append(
            Chunk(path, "<module>", "module", 1, _end(tree.body[0]) if tree.body else 1,
                  signature=f"module {path}", doc=mod_doc, ids=top, is_test=test)
        )

    def visit(body: list[ast.stmt], prefix: str, in_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qual = prefix + node.name
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                ids, strings = _walk_body(node.body, skip_first_doc=True)
                sig = _signature(node)
                decs = _decorators(node)
                if decs:
                    sig = " ".join("@" + d for d in decs) + " " + sig
                chunks.append(
                    Chunk(path, qual, "method" if in_class else "function", start, _end(node),
                          signature=sig, doc=ast.get_docstring(node) or "", ids=ids,
                          strings=strings, comments=comment_text(start, _end(node)), is_test=test)
                )
            elif isinstance(node, ast.ClassDef):
                qual = prefix + node.name
                header_end = min([n.lineno for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                                 + [_end(node) + 1]) - 1
                ids = [ast.unparse(b) for b in node.bases]
                for stmt in node.body:
                    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ids.append(stmt.name)
                    elif isinstance(stmt, ast.Assign):
                        ids += [t.id for t in stmt.targets if isinstance(t, ast.Name)]
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        ids.append(stmt.target.id)
                chunks.append(
                    Chunk(path, qual, "class", node.lineno, _end(node), signature=_signature(node),
                          doc=ast.get_docstring(node) or "", ids=ids,
                          comments=comment_text(node.lineno, header_end), is_test=test)
                )
                visit(node.body, qual + ".", True)

    visit(tree.body, "", False)
    return chunks
