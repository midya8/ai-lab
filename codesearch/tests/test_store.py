import json

from codesearch import chunker
from codesearch.store import discover, sync


def test_discover_skips_junk_dirs_and_large_files(repo):
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "x.py").write_text("def a():\n    pass\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "hook.py").write_text("x = 1\n")
    (repo / "pkg.egg-info").mkdir()
    (repo / "pkg.egg-info" / "y.py").write_text("x = 1\n")
    files = discover(repo)
    assert "app/retry.py" in files and not any("node_modules" in f or ".git" in f or "egg-info" in f for f in files)
    assert discover(repo, include=["app"]) == ["app/retry.py", "app/storage.py", "app/textutil.py"]
    assert discover(repo, include=["app/retry.py"]) == ["app/retry.py"]


def test_first_sync_parses_everything_second_reuses_all(repo):
    store = repo / ".codesearch" / "index.json"
    chunks, rep = sync(repo, store)
    assert len(rep.added) == 4 and rep.reused == 0 and store.exists()
    calls = []

    def spy(src, path):
        calls.append(path)
        return chunker.chunk_source(src, path)

    chunks2, rep2 = sync(repo, store, parse=spy)
    assert calls == [] and rep2.reused == 4 and [c.id for c in chunks2] == [c.id for c in chunks]


def test_only_changed_files_are_reparsed(repo):
    store = repo / "idx.json"
    sync(repo, store)
    (repo / "app" / "textutil.py").write_text("def only_one():\n    '''new'''\n    return 1\n")
    calls = []

    def spy(src, path):
        calls.append(path)
        return chunker.chunk_source(src, path)

    chunks, rep = sync(repo, store, parse=spy)
    assert calls == ["app/textutil.py"] and rep.changed == ["app/textutil.py"]
    assert "app/textutil.py::only_one" in {c.id for c in chunks}
    assert "app/textutil.py::slugify" not in {c.id for c in chunks}


def test_removed_files_leave_the_index(repo):
    store = repo / "idx.json"
    sync(repo, store)
    (repo / "app" / "storage.py").unlink()
    chunks, rep = sync(repo, store)
    assert rep.removed == ["app/storage.py"]
    assert not any(c.path == "app/storage.py" for c in chunks)


def test_unparsable_file_is_reported_and_stays_reported(repo):
    store = repo / "idx.json"
    (repo / "app" / "broken.py").write_text("def x(:\n")
    chunks, rep = sync(repo, store)
    assert "app/broken.py" in rep.failed
    _, rep2 = sync(repo, store)
    assert "app/broken.py" in rep2.failed and rep2.reused == 5
    assert len(chunks) > 0


def test_corrupt_or_old_store_is_rebuilt(repo):
    store = repo / "idx.json"
    store.write_text("{not json")
    _, rep = sync(repo, store)
    assert len(rep.added) == 4
    data = json.loads(store.read_text())
    data["version"] = 0
    store.write_text(json.dumps(data))
    _, rep = sync(repo, store)
    assert len(rep.added) == 4


def test_no_store_means_no_file(repo):
    sync(repo, None)
    assert not list(repo.glob("**/*.json"))
