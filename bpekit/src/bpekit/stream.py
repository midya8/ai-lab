"""Streaming decode: turn token ids into text one at a time without ever
emitting a broken character when a multi-byte UTF-8 sequence is split across
tokens (the usual bug when streaming LLM output)."""
from __future__ import annotations
import codecs


class StreamDecoder:
    def __init__(self, tok):
        self.tok = tok
        self._dec = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def push(self, token_id: int) -> str:
        """Return the text newly completed by this token ('' if mid-character)."""
        t = self.tok
        if token_id in t._inv_special:
            raw = t._inv_special[token_id].encode()
        else:
            raw = t.vocab[token_id]
        return self._dec.decode(raw)

    def flush(self) -> str:
        """Emit any dangling partial bytes (as U+FFFD) and reset."""
        out = self._dec.decode(b"", final=True)
        self._dec.reset()
        return out


def stream_decode(tok, ids):
    """Yield text pieces; their concatenation equals tok.decode(ids)."""
    d = StreamDecoder(tok)
    for i in ids:
        s = d.push(i)
        if s:
            yield s
    s = d.flush()
    if s:
        yield s
