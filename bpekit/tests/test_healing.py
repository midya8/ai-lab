import pytest
from bpekit.bpe import train_fast
from bpekit import heal_prompt, healed_continuation

TEXT = "http://example.com/a http://example.org/b see http://x.io/c " * 30


def test_prefix_and_ids_recompose_prompt():
    tok = train_fast(TEXT, 300)
    prompt = "see http:"
    ids, prefix, allowed = heal_prompt(tok, prompt)
    assert tok.decode(ids).encode() + prefix == prompt.encode()
    assert prefix and allowed


def test_allowed_all_start_with_prefix_and_include_original():
    tok = train_fast(TEXT, 300)
    full = tok.encode("see http:")
    ids, prefix, allowed = heal_prompt(tok, "see http:")
    assert full[-1] in allowed
    assert all(tok.vocab[i].startswith(prefix) for i in allowed)


def test_healing_enables_longer_token():
    tok = train_fast(TEXT, 320)
    ids, prefix, allowed = heal_prompt(tok, "see http:")
    longer = [i for i in allowed if len(tok.vocab[i]) > len(prefix)]
    assert longer  # e.g. the merged "://" style token is now reachable


def test_n_back_and_empty():
    tok = train_fast(TEXT, 300)
    assert heal_prompt(tok, "") == ([], b"", [])
    ids2, p2, _ = heal_prompt(tok, "see http:", n_back=2)
    ids1, p1, _ = heal_prompt(tok, "see http:", n_back=1)
    assert len(ids2) == len(ids1) - 1 and p2.endswith(p1)


def test_special_not_removed():
    tok = train_fast(TEXT, 300)
    ids, prefix, allowed = heal_prompt(tok, "abc" + next(iter(tok.specials)))
    assert prefix == b"" and allowed == []


def test_continuation_check():
    tok = train_fast(TEXT, 300)
    ids, prefix, _ = heal_prompt(tok, "see http:")
    gen = tok.encode("see http://example.com")[len(ids):]
    assert healed_continuation(tok, ids, prefix, gen) == "see http://example.com"
    with pytest.raises(ValueError):
        healed_continuation(tok, ids, prefix, tok.encode("zzz"))
