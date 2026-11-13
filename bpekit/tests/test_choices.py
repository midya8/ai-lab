import pytest
from bpekit.bpe import train_fast
from bpekit import ChoiceConstraint, constrained_pick

TEXT = "positive negative neutral positive mixed negative " * 40


def tk():
    return train_fast(TEXT, 300)


def test_first_allowed_tokens_start_choices():
    tok = tk()
    cc = ChoiceConstraint(tok, ["positive", "negative"])
    for i in cc.allowed():
        assert any(c.startswith(tok.vocab[i]) for c in cc.choices)
    assert tok.encode("positive")[0] in cc.allowed()


def test_every_canonical_path_is_accepted():
    tok = tk()
    for c in ["positive", "negative", "neutral"]:
        cc = ChoiceConstraint(tok, ["positive", "negative", "neutral"])
        for i in tok.encode(c):
            cc.push(i)
        assert cc.done and cc.result() == c


def test_byte_level_path_accepted():
    tok = tk()
    cc = ChoiceConstraint(tok, ["mixed"])
    for b in b"mixed":
        cc.push(b)
    assert cc.done


def test_violation_raises():
    tok = tk()
    cc = ChoiceConstraint(tok, ["positive"])
    with pytest.raises(ValueError):
        cc.push(ord("z"))


def test_prefix_choice_complete_not_done_and_eos():
    tok = tk()
    eos = 9999
    cc = ChoiceConstraint(tok, ["ab", "abc"], eos_id=eos)
    cc.push(ord("a")); cc.push(ord("b"))
    assert cc.complete and not cc.done
    assert eos in cc.allowed() and ord("c") in cc.allowed()
    cc.push(eos)
    assert cc.result() == "ab"


def test_constrained_pick_follows_scores():
    tok = tk()
    target = tok.encode("negative")

    def score(ids, allowed):
        s = {i: 0.0 for i in allowed}
        if len(ids) < len(target) and target[len(ids)] in s:
            s[target[len(ids)]] = 1.0
        return s

    out, ids = constrained_pick(tok, ["positive", "negative"], score)
    assert out == "negative"


def test_empty_choices_rejected():
    with pytest.raises(ValueError):
        ChoiceConstraint(tk(), [])
