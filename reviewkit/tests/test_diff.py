from reviewkit.diff import parse_diff

GIT = """diff --git a/app/x.py b/app/x.py
index 111..222 100644
--- a/app/x.py
+++ b/app/x.py
@@ -1,4 +1,5 @@
 import os
-old = 1
+new = 1
+extra = 2
 keep = 3
 tail = 4
@@ -20,2 +21,3 @@ def f():
 a = 1
+b = 2
 c = 3
diff --git a/img.png b/img.png
new file mode 100644
Binary files /dev/null and b/img.png differ
diff --git a/gone.py b/gone.py
deleted file mode 100644
--- a/gone.py
+++ /dev/null
@@ -1,2 +0,0 @@
-x = 1
-y = 2
diff --git a/old name.py b/new name.py
similarity index 90%
rename from old name.py
rename to new name.py
"""


def test_line_numbers_and_files():
    files = parse_diff(GIT)
    assert [f.path for f in files] == ["app/x.py", "img.png", "gone.py", "new name.py"]
    x = files[0]
    assert x.added() == {2: "new = 1", 3: "extra = 2", 22: "b = 2"}
    assert x.visible()[1] == "import os" and x.visible()[4] == "keep = 3"
    assert files[1].binary and files[1].status == "added"
    assert files[2].status == "deleted" and files[2].added() == {}
    assert files[3].status == "renamed" and files[3].old_path == "old name.py"


def test_removed_line_starting_with_dashes_is_not_a_header():
    d = "--- a/f.py\n+++ b/f.py\n@@ -1,2 +1,1 @@\n--- not a header\n keep\n"
    f = parse_diff(d)[0]
    assert f.path == "f.py"
    assert [l.kind for l in f.hunks[0].lines] == ["-", " "]
    assert f.hunks[0].lines[0].text == "-- not a header"


def test_plain_unified_diff_and_new_file():
    d = "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
    f = parse_diff(d)[0]
    assert f.status == "added" and f.added() == {1: "a", 2: "b"}


def test_no_newline_marker_and_single_line_hunks():
    d = "--- a/f\n+++ b/f\n@@ -1 +1 @@\n-a\n\\ No newline at end of file\n+b\n\\ No newline at end of file\n"
    f = parse_diff(d)[0]
    assert f.added() == {1: "b"}


def test_multiple_files_without_git_headers():
    d = "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-p\n+q\n"
    assert [f.path for f in parse_diff(d)] == ["a.py", "b.py"]


def test_blank_context_line_without_leading_space():
    d = "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,3 +1,3 @@\n a\n\n-b\n+c\n"
    f = parse_diff(d)[0]
    assert f.added() == {3: "c"}


def test_empty_input():
    assert parse_diff("") == []
