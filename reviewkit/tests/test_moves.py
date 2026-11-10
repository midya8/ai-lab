from reviewkit.diff import parse_diff
from reviewkit.moves import detect_moves, moved_new_lines
from reviewkit.review import Config, Reviewer

BLOCK = ["def load(path):", "    data = open(path).read()", "    rows = data.split(chr(10))", "    return rows"]


def mk(old_lines, new_lines, old_path="a.py", new_path="b.py"):
    rem = "\n".join("-" + l for l in old_lines)
    add = "\n".join("+" + l for l in new_lines)
    return (f"diff --git a/{old_path} b/{old_path}\n--- a/{old_path}\n+++ b/{old_path}\n"
            f"@@ -1,{len(old_lines)} +1,0 @@\n{rem}\n"
            f"diff --git a/{new_path} b/{new_path}\n--- a/{new_path}\n+++ b/{new_path}\n"
            f"@@ -0,0 +1,{len(new_lines)} @@\n{add}\n")


def test_detects_verbatim_move_across_files():
    mv = detect_moves(parse_diff(mk(BLOCK, BLOCK)))
    assert len(mv) == 1
    assert (mv[0].from_path, mv[0].to_path, mv[0].length) == ("a.py", "b.py", 4)
    assert moved_new_lines(mv) == {"b.py": {1, 2, 3, 4}}


def test_indentation_change_still_a_move():
    mv = detect_moves(parse_diff(mk(BLOCK, ["    " + l for l in BLOCK])))
    assert len(mv) == 1


def test_short_runs_and_boilerplate_are_not_moves():
    assert detect_moves(parse_diff(mk(BLOCK[:2], BLOCK[:2]))) == []
    triv = ["}", "", "}", "return", "pass"]
    assert detect_moves(parse_diff(mk(triv, triv))) == []


def test_edited_block_is_not_moved():
    edited = list(BLOCK)
    edited[1] = "    data = open(path, 'rb').read()"
    assert detect_moves(parse_diff(mk(BLOCK, edited)), min_lines=3) == []


def test_each_removed_line_used_once():
    mv = detect_moves(parse_diff(mk(BLOCK, BLOCK + BLOCK)))
    assert len(mv) == 1 and mv[0].length == 4


def test_min_lines_configurable():
    assert len(detect_moves(parse_diff(mk(BLOCK[:2], BLOCK[:2])), min_lines=2)) == 1


EVAL = ["def run(x):", "    y = eval(x)", "    z = y + 1", "    return z"]


def test_reviewer_skips_findings_on_moved_code_but_not_new_code():
    moved = Reviewer(Config()).review(mk(EVAL, EVAL))
    assert not any(f.rule.endswith("eval") or "eval" in f.message.lower() for f in moved.all_findings)
    assert any("moved code" in n for n in moved.notes)
    fresh = Reviewer(Config()).review(mk(["x = 1"], EVAL))
    assert any("eval" in f.message.lower() for f in fresh.all_findings)
    off = Reviewer(Config(skip_moved=False)).review(mk(EVAL, EVAL))
    assert any("eval" in f.message.lower() for f in off.all_findings)
