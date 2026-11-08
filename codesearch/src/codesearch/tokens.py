"""Turning identifiers and English into comparable terms.

Code search lives or dies on this step: ``parseHTTPResponse``, ``parse_http_response``
and the query "parse an http response" must all land on the same terms.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_PIECE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

# Common English plus words that appear in nearly every Python function.
STOPWORDS = frozenset(
    """
    a an and any are as at be been being but by can could do does for from had has have how
    i if in into is it its of on or our should so than that the their them then there these
    they this those to was we were what when where which while who why will with would you your
    function method class code find show get give return returns returned using use used
    self cls none true false def import lambda pass else elif try except finally raise yield
    not also just over per via
    """.split()
)


def split_identifier(name: str) -> list[str]:
    """Split snake_case, camelCase, PascalCase, acronyms and digits into lowercase words.

    >>> split_identifier("parseHTTPResponse_v2")
    ['parse', 'http', 'response', 'v', '2']
    """
    out: list[str] = []
    for chunk in name.split("_"):
        out.extend(p.lower() for p in _PIECE_RE.findall(chunk))
    return out


def stem(word: str) -> str:
    """A deliberately small suffix stripper: enough to equate parse/parsing/parsed/parses.

    Not Porter. It never produces an empty word and leaves short words alone.
    """
    w = word
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith(("ches", "shes", "xes", "zes")):
        w = w[:-2]
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def terms(text: str, keep_stopwords: bool = False) -> list[str]:
    """Identifier-aware tokenisation of any text (code identifiers or prose)."""
    out: list[str] = []
    for word in _WORD_RE.findall(text):
        for piece in split_identifier(word):
            if piece.isdigit() or len(piece) < 2:
                continue
            if not keep_stopwords and piece in STOPWORDS:
                continue
            out.append(stem(piece))
    return out


def looks_like_symbol(token: str) -> bool:
    """Does a raw query token look like an identifier someone typed on purpose?"""
    t = token.strip("`()'\".,;:")
    if not t or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", t):
        return False
    last = t.split(".")[-1]
    if "_" in last.strip("_"):
        return True
    if re.search(r"[a-z][A-Z]", last):
        return True
    return token.startswith("`") or token.endswith("()")
