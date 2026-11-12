"""Token-budget truncation that never cuts a character in half."""


def _clean_prefix(tok, ids):
    """Longest prefix of ids that decodes without a replacement char."""
    ids = list(ids)
    while ids and "\ufffd" in tok.decode(ids) and "\ufffd" not in tok.decode(ids[:-1] or []) :
        ids.pop()
    return ids


def truncate(tok, text: str, max_tokens: int, keep: str = "head", marker: str = " … "):
    """Fit text into max_tokens. keep: 'head', 'tail' or 'middle' (head+tail
    joined by marker). Returns (text, n_tokens_of_result). Texts that already
    fit are returned unchanged. The result never exceeds max_tokens when
    re-encoded (marker cost is included for 'middle')."""
    if keep not in ("head", "tail", "middle"):
        raise ValueError("keep must be head, tail or middle")
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    ids = tok.encode(text)
    if len(ids) <= max_tokens:
        return text, len(ids)

    def head(n):
        return tok.decode(_clean_prefix(tok, ids[:n])) if n > 0 else ""

    def tail(n):
        if n <= 0:
            return ""
        t = ids[-n:]
        while t and tok.decode(t).startswith("\ufffd"):
            t = t[1:]
        return tok.decode(t)

    if keep == "head":
        out = head(max_tokens)
    elif keep == "tail":
        out = tail(max_tokens)
    else:
        m = len(tok.encode(marker))
        room = max(max_tokens - m, 0)
        out = head((room + 1) // 2) + marker + tail(room // 2)
    # re-encoding merges can differ at joins; shrink until it fits
    while len(tok.encode(out)) > max_tokens and out:
        out = out[:-1] if keep != "tail" else out[1:]
    return out, len(tok.encode(out))
