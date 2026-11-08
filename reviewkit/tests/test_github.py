import json
import re

import pytest

from reviewkit.bench import make_diff
from reviewkit.github import build_review, existing_fingerprints, post_review
from reviewkit.review import Reviewer


def result(src="x = eval(s)\ny = eval(t)\n"):
    return Reviewer().review(make_diff({"a.py": (None, src)}))


class FakeGitHub:
    """Records calls; `existing` is the list of review comments returned for GET."""

    def __init__(self, existing=None, post_codes=(200,)):
        self.existing = existing or []
        self.post_codes = list(post_codes)
        self.posts = []

    def __call__(self, method, url, token, body=None):
        assert token == "tok"
        if method == "GET":
            page = int(re.search(r"[&?]page=(\d+)", url).group(1))
            return 200, self.existing[(page - 1) * 100: page * 100]
        self.posts.append(body)
        return self.post_codes.pop(0), {"msg": "x"}


def test_build_review_shape():
    p = build_review(result(), commit_id="abc123")
    assert p["event"] == "COMMENT" and p["commit_id"] == "abc123"
    assert [(c["path"], c["line"], c["side"]) for c in p["comments"]] == [("a.py", 1, "RIGHT"), ("a.py", 2, "RIGHT")]
    assert "reviewkit:fp=" in p["comments"][0]["body"] and p["body"].startswith("### reviewkit reviewed 1 file")


def test_posts_inline_review():
    gh = FakeGitHub()
    out = post_review(result(), "o/r", 7, "tok", http=gh)
    assert out == {"posted": 2, "skipped": 0, "mode": "inline"} and len(gh.posts) == 1


def test_second_run_skips_already_posted_comments_even_after_line_shift():
    first = build_review(result())
    gh = FakeGitHub(existing=[{"body": c["body"]} for c in first["comments"]])
    out = post_review(result("# new header\nx = eval(s)\ny = eval(t)\n"), "o/r", 7, "tok", http=gh)
    assert out["mode"] == "nothing-new" and out["skipped"] == 2 and gh.posts == []


def test_only_new_findings_are_posted():
    first = build_review(result("x = eval(s)\n"))
    gh = FakeGitHub(existing=[{"body": c["body"]} for c in first["comments"]])
    out = post_review(result("x = eval(s)\ny = eval(t)\n"), "o/r", 7, "tok", http=gh)
    assert out == {"posted": 1, "skipped": 1, "mode": "inline"}
    assert [c["line"] for c in gh.posts[0]["comments"]] == [2]


def test_pagination_of_existing_comments():
    first = build_review(result())
    fp_body = first["comments"][0]["body"]
    gh = FakeGitHub(existing=[{"body": "noise"}] * 100 + [{"body": fp_body}])
    assert len(existing_fingerprints("o/r", 1, "tok", gh)) == 1


def test_422_falls_back_to_body_only_review():
    gh = FakeGitHub(post_codes=(422, 200))
    out = post_review(result(), "o/r", 7, "tok", http=gh)
    assert out["mode"] == "body-only" and "comments" not in gh.posts[1]
    assert "`a.py:1`" in gh.posts[1]["body"]


def test_other_failures_raise():
    with pytest.raises(RuntimeError, match="HTTP 403"):
        post_review(result(), "o/r", 7, "tok", http=FakeGitHub(post_codes=(403,)))


def test_clean_review_still_posts_summary():
    gh = FakeGitHub()
    out = post_review(Reviewer().review(make_diff({"a.py": (None, "x = 1\n")})), "o/r", 1, "tok", http=gh)
    assert out["mode"] == "inline" and gh.posts[0]["comments"] == []
    assert "found nothing" in gh.posts[0]["body"]


def test_listing_failure_raises():
    def bad(method, url, token, body=None):
        return 500, {}
    with pytest.raises(RuntimeError, match="could not list"):
        post_review(result(), "o/r", 1, "tok", http=bad)
