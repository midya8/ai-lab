"""Post a review to a pull request, idempotently.

Each inline comment carries `<!-- reviewkit:fp=... -->` fingerprints (hash of rule + the
line's text, not its number). Re-running on a new push skips comments already posted.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Callable, Dict, List, Optional, Set, Tuple

from .review import MARK, ReviewResult, group_by_line, render_line_comment, render_summary

Http = Callable[[str, str, str, Optional[dict]], Tuple[int, object]]
API = "https://api.github.com"


def default_http(method: str, url: str, token: str, body: Optional[dict] = None) -> Tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "reviewkit", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
            return resp.status, json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


def build_review(result: ReviewResult, commit_id: Optional[str] = None, skip: Optional[Set[str]] = None,
                 event: str = "COMMENT") -> dict:
    """The JSON body for POST /pulls/{n}/reviews. Findings whose fingerprints are in `skip` are left out."""
    skip = skip or set()
    comments = []
    for (path, line), fs in sorted(group_by_line(result.findings).items()):
        fs = [f for f in fs if f.fp not in skip]
        if fs:
            comments.append({"path": path, "line": line, "side": "RIGHT", "body": render_line_comment(fs)})
    payload = {"event": event, "body": render_summary(result), "comments": comments}
    if commit_id:
        payload["commit_id"] = commit_id
    return payload


def existing_fingerprints(repo: str, pr: int, token: str, http: Http = default_http) -> Set[str]:
    fps: Set[str] = set()
    page = 1
    while True:
        status, data = http("GET", f"{API}/repos/{repo}/pulls/{pr}/comments?per_page=100&page={page}", token, None)
        if status != 200 or not isinstance(data, list):
            raise RuntimeError(f"could not list existing review comments (HTTP {status})")
        for c in data:
            for m in re.finditer(re.escape(MARK) + r"([0-9a-f,]+)", c.get("body", "")):
                fps.update(m.group(1).split(","))
        if len(data) < 100:
            return fps
        page += 1


def post_review(result: ReviewResult, repo: str, pr: int, token: str, commit_id: Optional[str] = None,
                http: Http = default_http, event: str = "COMMENT") -> Dict[str, object]:
    """Post new findings. Returns {'posted': n, 'skipped': n, 'mode': 'inline'|'body-only'|'nothing-new'}."""
    seen = existing_fingerprints(repo, pr, token, http)
    payload = build_review(result, commit_id, seen, event)
    skipped = sum(1 for f in result.findings if f.fp in seen)
    if not payload["comments"] and result.findings:
        return {"posted": 0, "skipped": skipped, "mode": "nothing-new"}
    url = f"{API}/repos/{repo}/pulls/{pr}/reviews"
    status, data = http("POST", url, token, payload)
    if status in (200, 201):
        return {"posted": len(payload["comments"]), "skipped": skipped, "mode": "inline"}
    if status == 422 and payload["comments"]:
        # A line fell outside the diff view: keep the review by folding findings into the body.
        lines = [payload["body"], "", "Inline placement failed, so the findings are listed here:"]
        for f in result.findings:
            if f.fp not in seen:
                lines.append(f"- `{f.path}:{f.line}` **{f.severity}** {f.message}")
        fallback = {"event": event, "body": "\n".join(lines)}
        if commit_id:
            fallback["commit_id"] = commit_id
        status2, _ = http("POST", url, token, fallback)
        if status2 in (200, 201):
            return {"posted": len(payload["comments"]), "skipped": skipped, "mode": "body-only"}
        status = status2
    raise RuntimeError(f"GitHub rejected the review (HTTP {status}): {str(data)[:200]}")
