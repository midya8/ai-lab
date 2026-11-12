"""Sequence packing: concatenate documents into fixed-length training rows."""


def pack(tok, docs, seq_len: int, eos: str | None = None, drop_last: bool = True):
    """Pack docs into rows of exactly seq_len token ids (last row padded-free).

    Documents are joined by the `eos` special (default: first special, if any).
    Returns (rows, doc_starts): rows is a list of id lists; doc_starts[i] lists
    the in-row offsets where a document begins in row i.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if eos is None:
        eos_id = next(iter(tok.specials.values()), None)
    else:
        eos_id = tok.specials[eos]
    stream, starts = [], []
    for d in docs:
        starts.append(len(stream))
        stream.extend(tok.encode(d))
        if eos_id is not None:
            stream.append(eos_id)
    rows, doc_starts = [], []
    n = len(stream) if not drop_last else len(stream) - len(stream) % seq_len
    for off in range(0, n, seq_len):
        rows.append(stream[off:off + seq_len])
        doc_starts.append([s - off for s in starts if off <= s < off + seq_len])
    return rows, doc_starts
