import numpy as np

from voicekit import synth
from voicekit.segmenter import SegmenterConfig, StreamSegmenter, segment_signal


def utterance(snr=20, seed=0, pron="naiis", kind="white"):
    rng = np.random.default_rng(seed)
    w = synth.synth_word(pron, synth.random_speaker(rng), rng)
    x, region = synth.make_utterance(w, rng)
    return synth.add_noise(x, snr, rng, kind, region=region), region


def test_silence_and_pure_noise_yield_nothing():
    rng = np.random.default_rng(0)
    assert segment_signal(np.zeros(16000)) == []
    assert segment_signal(0.01 * rng.standard_normal(80000)) == []


def test_word_is_found_with_sane_boundaries():
    x, (a, b) = utterance(20)
    segs = segment_signal(x)
    assert len(segs) == 1
    s = segs[0]
    assert abs(s.start_s - a / 8000) < 0.15 and abs(s.end_s - b / 8000) < 0.2
    assert s.power.shape[0] == s.end_frame - s.start_frame and not s.truncated
    assert s.noise_psd.shape == (129,)


def test_two_words_with_a_gap_are_two_segments():
    x1, _ = utterance(20, 1)
    x2, _ = utterance(20, 2, "muisi")
    segs = segment_signal(np.concatenate([x1, np.zeros(8000), x2]))
    assert len(segs) == 2 and segs[0].end_frame < segs[1].start_frame


def test_short_gap_inside_a_word_is_bridged_by_hangover():
    # 'stop' contains a 50 ms closure; it must stay one segment
    segs = segment_signal(utterance(25, 3, "s-ou")[0])
    assert len(segs) == 1


def test_chunking_does_not_change_segments():
    x, _ = utterance(10, 4)
    x = np.concatenate([x, utterance(10, 5, "muisi")[0]])
    ref = [(s.start_frame, s.end_frame) for s in segment_signal(x)]
    for chunk in (1, 77, 160, 801, 5000):
        seg = StreamSegmenter()
        got = []
        for i in range(0, len(x), chunk):
            got += seg.feed(x[i:i + chunk])
        got += seg.flush()
        assert [(s.start_frame, s.end_frame) for s in got] == ref, chunk


def test_flush_closes_an_open_segment():
    x, (a, b) = utterance(25, 6)
    seg = StreamSegmenter()
    assert seg.feed(x[: b - 400]) == []  # word has not ended yet
    out = seg.flush()
    assert len(out) == 1


def test_too_short_blips_are_dropped():
    rng = np.random.default_rng(0)
    x = 0.001 * rng.standard_normal(16000)
    x[8000:8400] += 0.5 * rng.standard_normal(400)  # 50 ms burst
    assert segment_signal(x) == []


def test_sustained_loud_sound_is_truncated_then_absorbed():
    rng = np.random.default_rng(0)
    n = 8000 * 8
    x = 0.01 * rng.standard_normal(n)
    x[8000:] *= 30  # background level jumps and stays
    segs = segment_signal(x)
    assert segs and segs[0].truncated
    assert len(segs) <= 2  # floor re-estimated: it does not retrigger every 1.5 s forever


def test_noise_floor_does_not_chase_speech():
    # a long word must not raise the floor enough to cut itself in half
    segs = segment_signal(utterance(15, 7, "uese")[0])
    assert len(segs) == 1


def test_config_changes_sensitivity():
    x, _ = utterance(5, 8)
    strict = segment_signal(x, SegmenterConfig(on_margin=5.0, off_margin=4.0))
    assert strict == []


def test_score_segments_clean_word_has_high_iou():
    from voicekit.segmenter import score_segments
    x, region = utterance(20)
    r = score_segments(segment_signal(x), [region])
    assert r["recall"] == 1.0 and r["precision"] == 1.0
    assert r["mean_iou"] > 0.6 and r["onset_err_s"] < 0.15


def test_score_segments_edge_cases():
    from voicekit.segmenter import Segment, score_segments
    assert score_segments([], [(0, 8000)])["recall"] == 0.0
    assert score_segments([], [])["recall"] == 1.0
    seg = Segment(0, 100, np.zeros((100, 129)), np.zeros(129))  # 0.0-1.0 s
    r = score_segments([seg], [(0, 8000), (40000, 48000)])
    assert r["recall"] == 0.5 and r["precision"] == 1.0 and abs(r["mean_iou"] - 1.0) < 1e-9
    assert score_segments([seg], [(80000, 88000)])["precision"] == 0.0
