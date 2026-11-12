import numpy as np
import pytest

from voicekit import CLASSES, WORDS, Recognizer, build_dataset, render_stream, score_events, synth, train_model
from voicekit.pipeline import UNKNOWN, Event, render_example


@pytest.fixture(scope="module")
def model():
    return train_model(50, seed=0, epochs=50)


def test_classes_layout():
    assert CLASSES[:-1] == WORDS and CLASSES[UNKNOWN] == "_unknown" and len(CLASSES) == 9


def test_build_dataset_shapes_and_labels():
    X, y, snr, missed = build_dataset(4, 1, snr_range=(25, 30))
    assert X.shape == (len(y), 261) and len(snr) == len(y) and missed == 0
    assert set(y) == set(range(9)) and ((snr >= 25) & (snr <= 30)).all()


def test_build_dataset_is_reproducible_and_sequences_align():
    a = build_dataset(3, 5, with_sequences=True)
    b = build_dataset(3, 5, with_sequences=True)
    assert np.allclose(a[0], b[0]) and len(a[4]) == len(a[1])
    assert a[4][0].shape[1] == 13


def test_render_example_unknown_uses_requested_pool():
    x = render_example(UNKNOWN, np.random.default_rng(0), 20, oov=synth.OOV_TEST)
    assert np.isfinite(x).all()


def test_recognizer_labels_clean_words(model):
    rng = np.random.default_rng(100)
    right = 0
    for i, w in enumerate(WORDS * 3):
        x = render_example(i % 8, rng, 25)
        ev = Recognizer(model).process(x)
        right += len(ev) == 1 and ev[0].word == WORDS[i % 8]
    assert right / 24 >= 0.85


def test_recognizer_events_are_chunk_independent(model):
    rng = np.random.default_rng(3)
    sig, _ = render_stream(["lights", "on", "weather"], rng, 15)
    ref = [(e.word, round(e.start_s, 2)) for e in Recognizer(model).process(sig, chunk=800)]
    for chunk in (1, 333, 4000):
        got = [(e.word, round(e.start_s, 2)) for e in Recognizer(model).process(sig, chunk=chunk)]
        assert got == ref


def test_recognizer_rejects_non_speech_bursts(model):
    rng = np.random.default_rng(4)
    rejected = 0
    for _ in range(15):
        b = synth.noise_burst(rng)
        x = synth.add_noise(np.concatenate([np.zeros(6000), b, np.zeros(6000)]), 20, rng, "white", region=(6000, 6000 + len(b)))
        rejected += len(Recognizer(model).process(x)) == 0
    assert rejected >= 12


def test_high_threshold_rejects_everything_with_reason(model):
    rng = np.random.default_rng(5)
    rec = Recognizer(model, threshold=1.01)
    assert rec.process(render_example(0, rng, 25)) == []
    assert rec.rejected and rec.rejected[0][1] in {"low_confidence", "unknown"}


def test_render_stream_truth_matches_signal_and_noise_level():
    rng = np.random.default_rng(0)
    sig, truth = render_stream(["on", "off"], rng, 40)
    assert [t[0] for t in truth] == ["on", "off"] and truth[0][1] == pytest.approx(1.0)
    assert truth[1][1] > truth[0][2]
    assert len(sig) / synth.SR > truth[1][2]
    quiet = sig[: synth.SR // 2]
    assert 0.0005 < quiet.std() < 0.002  # 40 dB below nominal 0.1 speech RMS


def test_score_events_counts():
    truth = [("on", 1.0, 1.5), ("off", 3.0, 3.5), ("stop", 5.0, 5.5)]
    ev = [Event("on", 1, 1.05, 1.5), Event("stop", 1, 3.0, 3.5), Event("play", 1, 8.0, 8.4)]
    s = score_events(ev, truth)
    assert (s["tp"], s["fp"], s["fn"], s["wrong_word"]) == (1, 2, 2, 1)
    assert s["precision"] == pytest.approx(1 / 3) and s["recall"] == pytest.approx(1 / 3)


def test_score_events_perfect_and_empty():
    truth = [("on", 1.0, 1.5)]
    assert score_events([Event("on", 1, 1.0, 1.5)], truth)["f1"] == 1.0
    s = score_events([], truth)
    assert s["recall"] == 0.0 and s["f1"] == 0.0


def test_score_events_needs_half_overlap_and_no_double_use():
    truth = [("on", 1.0, 2.0)]
    assert score_events([Event("on", 1, 1.8, 2.5)], truth)["tp"] == 0
    s = score_events([Event("on", 1, 1.0, 1.6), Event("on", 1, 1.5, 2.0)], truth)
    assert s["tp"] == 1 and s["fp"] == 1


def test_calibrate_thresholds_reaches_target_on_validation(model):
    from voicekit import calibrate_thresholds
    X, y, _, _ = build_dataset(20, 77)
    thr = calibrate_thresholds(model, X, y, target_precision=0.9)
    assert set(thr) == set(WORDS) and all(0.3 <= v <= 0.99 for v in thr.values())
    P = model.predict_proba(X)
    pred, conf = P.argmax(1), P.max(1)
    for k, w in enumerate(WORDS):
        sel = (pred == k) & (conf >= thr[w])
        if thr[w] < 0.99 and sel.sum() >= 3:
            assert (y[sel] == k).mean() >= 0.9


def test_calibrate_unreachable_target_falls_back_to_max(model):
    from voicekit import calibrate_thresholds
    X, y, _, _ = build_dataset(5, 78)
    thr = calibrate_thresholds(model, X, np.full_like(y, UNKNOWN), target_precision=0.99)
    assert all(v == 0.99 for v in thr.values())


def test_recognizer_accepts_per_word_threshold_dict(model):
    x = render_example(0, np.random.default_rng(5), 25)
    assert len(Recognizer(model, threshold={w: 0.0 for w in WORDS}).process(x)) == 1
    r = Recognizer(model, threshold={w: 1.01 for w in WORDS})
    assert r.process(x) == [] and any(why == "low_confidence" for _, why in r.rejected)
