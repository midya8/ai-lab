import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import pytest
from ragevals.faithfulness import faithfulness, answer_relevance, evaluate_answers, split_claims
from ragevals.cli import main

ROOT = pathlib.Path(__file__).resolve().parents[1]
CTX = ["LoRA freezes the pretrained model weights and injects trainable low-rank matrices into each layer."]


def test_supported_answer_scores_one():
    assert faithfulness("LoRA freezes pretrained weights and trains low-rank matrices.", CTX)["score"] == 1.0


def test_hallucinated_claim_flagged():
    r = faithfulness("LoRA freezes pretrained weights. It was invented by Google in 2015.", CTX)
    assert r["score"] == 0.5 and r["unsupported"] == ["It was invented by Google in 2015."]


def test_custom_judge_and_empty():
    assert faithfulness("Anything goes here today.", CTX, supports=lambda c, ctx: False)["score"] == 0.0
    assert faithfulness("", CTX)["score"] == 1.0
    assert split_claims("Hi. Real claim about transformers attention.") == ["Real claim about transformers attention."]


def test_relevance_and_aggregate():
    assert answer_relevance("How does LoRA reduce parameters?", "LoRA reduces nothing") < 1
    with pytest.raises(ValueError):
        evaluate_answers([])


def test_cli_bundled_answers(capsys):
    assert main(["--faithfulness", str(ROOT / "data" / "answers.json")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n"] == 2 and out["hallucinated_answers"] == 1 and 0 < out["faithfulness"] < 1
