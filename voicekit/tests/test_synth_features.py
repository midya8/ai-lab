import numpy as np
import pytest

from voicekit import synth
from voicekit import features as F


def rng(s=0):
    return np.random.default_rng(s)


def test_every_word_synthesizes_with_target_rms():
    for pron in list(synth.VOCAB.values()) + list(synth.OOV_TRAIN.values()) + list(synth.OOV_TEST.values()):
        w = synth.synth_word(pron, synth.random_speaker(rng(1)), rng(2))
        active = np.abs(w) > 0.02 * np.abs(w).max()
        assert np.isfinite(w).all() and len(w) > 800
        assert np.sqrt(np.mean(w[active] ** 2)) == pytest.approx(0.1, rel=1e-6)


def test_synthesis_is_deterministic_and_speaker_dependent():
    sp = synth.Speaker(120, 1.0, 1.0)
    a = synth.synth_word("naiis", sp, rng(5))
    b = synth.synth_word("naiis", sp, rng(5))
    c = synth.synth_word("naiis", synth.Speaker(200, 1.1, 1.0), rng(5))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_rate_changes_duration():
    slow = synth.synth_word("naiis", synth.Speaker(120, 1.0, 0.8), rng(1))
    fast = synth.synth_word("naiis", synth.Speaker(120, 1.0, 1.25), rng(1))
    assert len(slow) > 1.4 * len(fast)


def test_unknown_phoneme_and_empty_word_rejected():
    with pytest.raises(ValueError):
        synth.synth_word("xyz", synth.Speaker(120, 1, 1), rng())
    with pytest.raises(ValueError):
        synth.synth_word("", synth.Speaker(120, 1, 1), rng())


def test_wide_speakers_leave_the_training_tract_range():
    tracts = [synth.random_speaker(rng(i), wide=True).tract for i in range(50)]
    assert all(t < 0.92 or t > 1.12 for t in tracts)


@pytest.mark.parametrize("kind", ["white", "pink", "hum"])
@pytest.mark.parametrize("snr", [0, 10, 20])
def test_add_noise_hits_requested_snr(kind, snr):
    w = synth.synth_word("muisi", synth.random_speaker(rng(3)), rng(3))
    x, region = synth.make_utterance(w, rng(4))
    y = synth.add_noise(x, snr, rng(6), kind, region=region)
    noise = y - x
    got = 10 * np.log10(np.mean(x[region[0]:region[1]] ** 2) / np.mean(noise**2))
    assert got == pytest.approx(snr, abs=1.0)


def test_add_noise_rejects_unknown_kind():
    with pytest.raises(ValueError):
        synth.add_noise(np.ones(100), 10, rng(), "purple")


def test_noise_burst_is_short_and_finite():
    b = synth.noise_burst(rng(1))
    assert 0.1 * synth.SR < len(b) < 0.5 * synth.SR and np.isfinite(b).all()


def test_wav_roundtrip_and_resample(tmp_path):
    x = synth.synth_word("on".replace("n", "n"), synth.Speaker(120, 1, 1), rng())
    p = tmp_path / "a.wav"
    synth.write_wav(str(p), x)
    y = synth.read_wav(str(p))
    assert len(y) == len(x)
    assert np.corrcoef(x, y)[0, 1] > 0.999


def test_read_wav_resamples_from_16k(tmp_path):
    import wave
    t = np.arange(16000) / 16000
    sig = (np.sin(2 * np.pi * 440 * t) * 20000).astype("<i2")
    p = tmp_path / "b.wav"
    with wave.open(str(p), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(sig.tobytes())
    y = synth.read_wav(str(p))
    assert abs(len(y) - 8000) <= 1


# --- features ---------------------------------------------------------------------------

def test_mel_filterbank_properties():
    fb = F.mel_filterbank()
    assert fb.shape == (F.N_MELS, F.N_FFT // 2 + 1)
    assert (fb >= 0).all() and (fb.max(1) > 0).all()
    centres = fb.argmax(1)
    assert (np.diff(centres) > 0).all()  # monotone centre frequencies


def test_mel_hz_roundtrip():
    f = np.array([100.0, 1000.0, 3500.0])
    assert np.allclose(F.mel2hz(F.hz2mel(f)), f)


def test_dct_is_orthonormal():
    assert np.allclose(F._DCT @ F._DCT.T, np.eye(13), atol=1e-9)


def test_frame_power_matches_batch_framing():
    x = rng(1).standard_normal(3000)
    P = F.power_frames(x)
    assert P.shape[0] == 1 + (3000 - F.frame_len()) // F.hop_len()
    assert np.allclose(P[3], F.frame_power(x[3 * F.hop_len(): 3 * F.hop_len() + F.frame_len()]))


def test_short_signal_gives_no_frames():
    assert F.power_frames(np.zeros(10)).shape[0] == 0


def test_sine_energy_lands_in_the_right_bin():
    t = np.arange(2000) / synth.SR
    P = F.power_frames(np.sin(2 * np.pi * 1000 * t))
    assert abs(P[3].argmax() * synth.SR / F.N_FFT - 1000) <= 40


def test_cmvn_zero_mean_unit_variance():
    Z = F.cmvn(rng(1).standard_normal((30, 13)) * 5 + 3)
    assert np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-4)


def test_cmvn_removes_gain():
    x = synth.synth_word("naiis", synth.Speaker(120, 1, 1), rng())
    a = F.cmvn(F.mfcc_from_power(F.power_frames(x)))
    b = F.cmvn(F.mfcc_from_power(F.power_frames(x * 7.0)))
    assert np.allclose(a, b, atol=1e-3)


def test_time_normalize_keeps_endpoints_and_length():
    A = np.arange(10.0)[:, None] * np.ones((1, 3))
    Z = F.time_normalize(A, 20)
    assert Z.shape == (20, 3) and np.allclose(Z[0], A[0]) and np.allclose(Z[-1], A[-1])
    assert (np.diff(Z[:, 0]) > 0).all()
    with pytest.raises(ValueError):
        F.time_normalize(np.zeros((0, 3)))
    assert F.time_normalize(np.ones((1, 3)), 5).shape == (5, 3)


def test_spectral_subtraction_floor_and_effect():
    P = np.abs(rng(1).standard_normal((10, 129))) + 1.0
    N = np.full(129, 0.9)
    out = F.spectral_subtract(P, N)
    assert (out >= 0.05 * P - 1e-12).all() and (out <= P).all()


def test_utterance_vector_has_fixed_length_regardless_of_duration():
    s = synth.Speaker(120, 1, 1)
    a = F.utterance_vector(F.power_frames(synth.synth_word("on", s, rng())))
    b = F.utterance_vector(F.power_frames(synth.synth_word("naiis", s, rng())))
    assert a.shape == b.shape == (20 * 13 + 1,)
    assert a[-1] < b[-1]  # log duration
