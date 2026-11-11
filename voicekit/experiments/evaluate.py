"""Full evaluation on synthetic formant speech. Run: PYTHONPATH=src python experiments/evaluate.py

Train speakers/utterances come from seed 0; every test set uses different seeds (so unseen
speakers and noise). The "unknown" class is trained on OOV_TRAIN words + noise bursts and
tested on OOV_TEST words, which the model never saw. Thresholds were fixed before testing.
"""
import json
import sys
import time

import numpy as np

from voicekit import (CLASSES, DTWClassifier, MLP, Recognizer, build_dataset, parse_events, render_stream,
                      score_events)
from voicekit.pipeline import UNKNOWN, WORDS
from voicekit import synth
from voicekit.synth import OOV_TEST, SR, VOCAB, random_speaker, synth_word

N_TRAIN, N_TEST, SNRS = 150, 30, (0, 5, 10, 20, 30)
THRESH = 0.6
out = {}
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


# ---- train -------------------------------------------------------------------------------
X, y, _, miss, seqs = build_dataset(N_TRAIN, 0, with_sequences=True)
Xn, yn, _, _ = build_dataset(N_TRAIN, 0, spectral_sub=False)
log(f"train: {len(y)} samples ({miss} missed by VAD)")
mlp = MLP(seed=0).fit(X, y, n_classes=len(CLASSES))
mlp_nosub = MLP(seed=0).fit(Xn, yn, n_classes=len(CLASSES))
dtw = DTWClassifier(per_class=5, seed=0).fit(seqs, y)
log("trained MLP, MLP (no spectral subtraction), DTW templates")

# ---- per-SNR test ------------------------------------------------------------------------
rows = []
total_per_snr = N_TEST * len(CLASSES)
for snr in SNRS:
    Xt, yt, _, ms, st = build_dataset(N_TEST, 5000 + snr, (snr, snr), oov=OOV_TEST, with_sequences=True)
    Xtn, ytn, _, _ = build_dataset(N_TEST, 5000 + snr, (snr, snr), oov=OOV_TEST, spectral_sub=False)
    p = mlp.predict_proba(Xt)
    pred = p.argmax(1)
    row = {
        "snr_db": snr,
        "vad_miss": ms / total_per_snr,
        "mlp_acc": float((pred == yt).mean()),
        "mlp_end_to_end": float((pred == yt).sum() / total_per_snr),
        "mlp_nosub_acc": float((mlp_nosub.predict(Xtn) == ytn).mean()),
        "dtw_acc": float(np.mean([dtw.predict(s, len(CLASSES)) == t for s, t in zip(st, yt)])),
    }
    known = yt != UNKNOWN
    row["known_accept_correct"] = float(((pred == yt) & (p.max(1) >= THRESH))[known].mean())
    row["unknown_rejected"] = float(((pred == UNKNOWN) | (p.max(1) < THRESH))[~known].mean())
    rows.append(row)
    log(row)
out["per_snr"] = rows

# ---- unseen vocal tracts -------------------------------------------------------------------
Xw, yw, _, mw, sw = build_dataset(N_TEST, 7000, (20, 20), wide=True, oov=OOV_TEST, with_sequences=True)
out["wide_speakers_20db"] = {
    "mlp_acc": float((mlp.predict(Xw) == yw).mean()),
    "dtw_acc": float(np.mean([dtw.predict(s, len(CLASSES)) == t for s, t in zip(sw, yw)])),
    "vad_missed": mw,
}
log("wide speakers", out["wide_speakers_20db"])

# ---- streaming word spotting ---------------------------------------------------------------
rec_model = mlp
stream_rows = []
for snr in (20, 10, 5):
    agg = {"tp": 0, "fp": 0, "fn": 0, "wrong_word": 0}
    rng = np.random.default_rng(9000 + snr)
    for k in range(12):
        words = [WORDS[i] for i in rng.integers(0, len(WORDS), 8)]
        sig, truth = render_stream(words, rng, snr, noise_kind=("white", "pink", "hum")[k % 3])
        ev = Recognizer(rec_model, threshold=THRESH).process(sig)
        s = score_events(ev, truth)
        for key in agg:
            agg[key] += s[key]
    prec = agg["tp"] / max(agg["tp"] + agg["fp"], 1)
    rec = agg["tp"] / (agg["tp"] + agg["fn"])
    agg.update(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    agg["snr_db"] = snr
    stream_rows.append(agg)
    log("stream", agg)
out["stream"] = stream_rows

# ---- false alarms on speech-free noise --------------------------------------------------
fa = {}
rng = np.random.default_rng(11)
for kind in ("white", "pink", "hum"):
    secs, events = 0.0, 0
    for _ in range(5):
        sig, _ = render_stream([], rng, 10, noise_kind=kind, lead=60.0, tail=0.0)
        events += len(Recognizer(rec_model, threshold=THRESH).process(sig))
        secs += len(sig) / 8000
    fa[kind] = {"seconds": secs, "false_events": events, "per_hour": events / secs * 3600}
out["noise_only"] = fa
log("noise-only", fa)

# ---- command (intent) accuracy ---------------------------------------------------------------
PHRASES = [(["lights", "on"], "lights.on"), (["lights", "off"], "lights.off"), (["play", "music"], "music.play"),
           (["music", "stop"], "music.stop"), (["timer", "on"], "timer.on"), (["weather"], "weather.query"),
           (["stop"], "stop_all"), (["off", "timer"], "timer.off")]
intent_rows = []
for snr in (20, 10):
    rng = np.random.default_rng(12000 + snr)
    ok = n = 0
    for _ in range(40):
        picks = [PHRASES[i] for i in rng.integers(0, len(PHRASES), 3)]
        words, truth_intents = [], []
        for ws, name in picks:
            words += ws
            truth_intents.append(name)
        # words inside a phrase are close together; phrases are separated by long silences
        sig_parts, truth = [np.zeros(SR)], []
        # built by hand so gaps differ inside vs between phrases
        pos = SR
        for ws, _ in picks:
            for j, w in enumerate(ws):
                wav = synth_word(VOCAB[w], random_speaker(rng), rng)
                truth.append((w, pos / SR, (pos + len(wav)) / SR))
                sig_parts.append(wav)
                g = np.zeros(int((rng.uniform(0.25, 0.5) if j < len(ws) - 1 else rng.uniform(2.2, 3.0)) * SR))
                sig_parts.append(g)
                pos += len(wav) + len(g)
        x = np.concatenate(sig_parts)
        ref = np.concatenate([np.full(1000, 0.1), np.zeros(len(x))])
        x = x + synth.add_noise(ref, snr, rng, "white", region=(0, 1000))[1000:]
        ev = Recognizer(rec_model, threshold=THRESH).process(x)
        got = [i.name if i else None for _, i in parse_events(ev, window_s=1.2)]
        n += len(truth_intents)
        ok += sum(a == b for a, b in zip(got, truth_intents)) if len(got) == len(truth_intents) else 0
    intent_rows.append({"snr_db": snr, "commands": n, "correct": ok, "command_accuracy": ok / n})
    log("intents", intent_rows[-1])
out["intents"] = intent_rows

json.dump(out, open("experiments/results.json", "w"), indent=2)
log("wrote experiments/results.json")
