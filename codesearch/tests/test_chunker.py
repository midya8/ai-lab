import pytest

from codesearch.chunker import chunk_source, is_test_path

SRC = '''"""Module doc."""
import os


# helper that doubles
def double(x: int) -> int:
    """Return twice x."""
    # multiply here
    return x * 2


@property
async def fetch(url, *, timeout=3):
    return await get(url, timeout=timeout)


class Box(Base):
    """A box."""
    size = 3
    label: str = "box"

    def open(self):
        """Open it."""
        def inner():
            return "nested secret"
        return inner()

    class Lid:
        def close(self):
            pass
'''


def by_name(chunks):
    return {c.qualname: c for c in chunks}


def test_chunks_functions_methods_classes_and_module():
    cs = by_name(chunk_source(SRC, "pkg/mod.py"))
    assert set(cs) == {"<module>", "double", "fetch", "Box", "Box.open", "Box.Lid", "Box.Lid.close"}
    assert cs["double"].kind == "function" and cs["Box.open"].kind == "method"
    assert cs["<module>"].kind == "module" and cs["Box"].kind == "class"


def test_line_ranges_and_ids():
    cs = by_name(chunk_source(SRC, "pkg/mod.py"))
    d = cs["double"]
    assert SRC.splitlines()[d.start - 1].startswith("def double")
    assert d.end > d.start
    assert d.id == "pkg/mod.py::double" and d.name == "double"
    assert cs["Box.Lid.close"].name == "close"


def test_decorators_start_the_chunk_and_enter_the_signature():
    f = by_name(chunk_source(SRC, "m.py"))["fetch"]
    assert SRC.splitlines()[f.start - 1].startswith("@property")
    assert f.signature.startswith("@property async def fetch(")


def test_signature_has_annotations():
    assert by_name(chunk_source(SRC, "m.py"))["double"].signature == "def double(x: int) -> int"


def test_docstring_excluded_from_body_but_nested_strings_kept():
    cs = by_name(chunk_source(SRC, "m.py"))
    assert cs["double"].doc == "Return twice x."
    assert "Return twice x." not in cs["double"].strings
    assert "nested secret" in cs["Box.open"].strings
    assert "inner" in cs["Box.open"].ids


def test_comments_are_attached_to_the_enclosing_chunk_only():
    cs = by_name(chunk_source(SRC, "m.py"))
    assert "multiply here" in cs["double"].comments
    assert "multiply here" not in cs["Box.open"].comments


def test_class_chunk_lists_members_without_method_bodies():
    box = by_name(chunk_source(SRC, "m.py"))["Box"]
    assert {"open", "size", "label", "Base"} <= set(box.ids)
    assert "inner" not in box.ids


def test_no_module_chunk_without_docstring():
    assert not any(c.kind == "module" for c in chunk_source("def f():\n    pass\n", "x.py"))


def test_syntax_error_propagates():
    with pytest.raises(SyntaxError):
        chunk_source("def broken(:\n", "bad.py")


def test_test_path_detection():
    assert is_test_path("tests/test_x.py") and is_test_path("pkg/test_x.py")
    assert is_test_path("a/x_test.py") and is_test_path("conftest.py")
    assert not is_test_path("src/contest.py") and not is_test_path("src/latest.py")


def test_chunk_dict_round_trip():
    from codesearch.chunker import Chunk
    c = by_name(chunk_source(SRC, "m.py"))["double"]
    assert Chunk.from_dict(c.to_dict()) == c
