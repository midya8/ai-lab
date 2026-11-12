# voicekit

A small, fully offline **voice-command pipeline built from scratch on NumPy**: streaming voice-activity
detection, MFCC features, two word classifiers (a hand-backpropped MLP and a DTW template matcher),
a streaming word spotter with open-set rejection, and a tiny order-tolerant command grammar.
No audio libraries, no model downloads, no API keys.

```
 audio chunks ──► StreamSegmenter ──► MFCC (+ spectral subtraction) ──► classifier ──► Event(word, p)
 (any chunk size)   adaptive noise floor      per-segment noise snapshot    MLP / DTW       │
                    hysteresis + hangover                                  "_unknown" class │
                                                                           + p threshold    ▼
                                                                          group_events ► parse_words ► Intent
```

> **Read this first.** There is no microphone data here. Speech is *synthesised*: each word is a
> sequence of vowel/nasal/fricative phonemes rendered as a harmonic source shaped by two gliding
> formant resonances (`synth.py`), with random speaker pitch, vocal-tract length and speaking rate,
> plus white / pink / mains-hum noise at a chosen SNR. It is a controlled testbed for the *engineering*
> (VAD, streaming, open-set rejection, evaluation protocol). **The numbers below say nothing about
> accuracy on human voices**, and a model trained here will not work on real speech.
> The segmenter (`python -m voicekit segment file.wav`) does run on any 16-bit WAV, but is untuned for real recordings.

## Quick start

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m voicekit demo          # trains a model (~15 s) and recognises a noisy command stream
PYTHONPATH=src python -m voicekit train --out model.npz
PYTHONPATH=src python -m voicekit synth lights --out lights.wav --snr 10
PYTHONPATH=src python -m voicekit segment lights.wav
PYTHONPATH=src python -m pytest -q              # 81 tests, ~5 s
PYTHONPATH=src python experiments/evaluate.py   # the full evaluation below, ~1 min
```

Demo output (seed 4, 10 dB white noise; the speech said is `lights on / play music / weather / timer on / stop`):

```
  0.97- 1.42s  lights   p=1.00
  1.73- 1.92s  on       p=1.00
  4.44- 4.80s  play     p=1.00
  5.10- 5.60s  music    p=1.00
  ...
  command [lights on] -> lights.on
  command [play music] -> music.play
  command [weather] -> weather.query
  command [timer on] -> timer.on
  command [stop] -> stop_all
```

Not every seed is that clean. With `--seed 3` one speaker says "play" so fast that only a 140 ms fragment
is segmented, and it is confidently labelled `lights` (p=0.98), so `play music` turns into
`lights` / *not understood*. That failure (a clipped short word that looks like another word) is real and
is what the confidence threshold alone cannot catch.

## Library

```python
import numpy as np
from voicekit import Recognizer, train_model, parse_events

model = train_model(n_per_class=150)             # synthetic training data, ~15 s
rec = Recognizer(model, threshold=0.6)
for chunk in audio_chunks:                       # any chunk size; results are identical
    for ev in rec.feed(chunk):
        print(ev.word, ev.confidence, ev.start_s, ev.end_s)
events = rec.flush()
for group, intent in parse_events(events):       # [lights, on] -> Intent("lights.on")
    ...
```

| Module | What it does |
|---|---|
| `synth.py` | Formant-style word synthesis, speakers, 3 noise types at exact SNR, noise bursts, WAV I/O |
| `features.py` | Mel filterbank, MFCC (DCT-II), CMVN, time normalisation, spectral subtraction |
| `segmenter.py` | Streaming VAD: chunk-size invariant, adaptive noise floor, hysteresis, hangover, pre-roll, truncation flag |
| `models.py` | `MLP` (manual backprop, Adam, dropout; gradient-checked in tests) and `DTWClassifier` (banded DTW, 1-NN) |
| `pipeline.py` | Dataset rendering through the real VAD, `Recognizer`, `render_stream`, event scoring |
| `intents.py` | Event grouping by time gap and an order-tolerant grammar (`off lights` = `lights off`) |

### The VAD score

Total band energy over the noise floor barely moves at low SNR, because speech power sits in a few
formant regions. The segmenter instead scores each frame by the mean of its **4 strongest mel bands'
log-excess over the running noise floor**. On noise-only frames this stays below ~0.6 (each mel band averages
many FFT bins); speech frames at 0-5 dB clear 0.95. Trigger at 0.95 for 3 frames, stay open above 0.70,
close after 150 ms below it. Those two margins were set by looking at noise-only and speech-frame score
distributions on a development seed, not tuned on the test seeds.

## Evaluation (`experiments/evaluate.py`, results in `experiments/results.json`)

Training: 150 rendered utterances per class (8 command words + `_unknown` made of 4 OOV words and non-speech
noise bursts), SNR uniform 0-30 dB, noise type random. Test: **different seeds** (unseen speakers and noise),
30 per class per SNR, and the unknown class uses **4 held-out OOV words** the model never saw. Everything goes through the
real VAD; utterances the VAD misses are counted as errors in the end-to-end column.

| SNR | VAD miss | MLP acc. (detected) | **MLP end-to-end** | MLP, no spectral sub. | DTW acc. (detected) | Known word accepted & right | Unknown rejected |
|---|---|---|---|---|---|---|---|
| 30 dB | 0% | 0.978 | **0.978** | 0.981 | 0.933 | 1.00 | 0.87 |
| 20 dB | 0% | 0.974 | **0.974** | 0.974 | 0.930 | 1.00 | 0.80 |
| 10 dB | 0.4% | 0.959 | **0.956** | 0.963 | 0.926 | 0.99 | 0.70 |
| 5 dB | 11% | 0.946 | **0.841** | 0.942 | 0.833 | 0.95 | 0.93 |
| 0 dB | 43% | 0.915 | **0.519** | 0.922 | 0.856 | 0.95 | 0.88 |

(270 test utterances per row; "detected" accuracies are over the utterances the VAD found, so the 0 dB row is
computed on far fewer samples than the others and is noisy. Threshold 0.6 was fixed before testing.)

What the table says, including what did not work:

- **At 10 dB and above the pipeline is dependable (96-98%); below 5 dB the bottleneck is the VAD, not the classifier.**
  Classifier accuracy on what the VAD does find stays above 0.9; at 0 dB the VAD loses 43% of words.
- **Spectral subtraction did not help.** The MLP is trained across 0-30 dB so it already sees noisy features, and the
  no-subtraction model matches or edges out the subtraction one at every SNR (differences are within one or two
  utterances). It is kept as an option (`spectral_sub=`); I would not claim a benefit from it here.
- **The MLP beats DTW (5 templates per class) at every SNR**, by 3-5 points at 10-30 dB and by 6-11 points at 0-5 dB.
  DTW needs no training, which is its only advantage here.
- **Open-set rejection is the weakest part.** 70-93% of held-out OOV words/bursts are rejected; the rest are
  confidently mapped to a vocabulary word. Held-out words such as `yellow` are phonetically close to `on`/`lights`
  in this synthetic phoneme space, which a real accent/phone inventory would not mirror, so do not read this as a rate.
- **Unseen vocal tracts:** with speakers whose tract scale lies *outside* the training range (0.78-0.90 and 1.14-1.28 vs
  0.92-1.12), at 20 dB MLP accuracy falls from 0.974 to **0.881** and DTW from 0.930 to 0.881 — they tie, so
  neither classifier generalises past its training range for free.

### Streaming (continuous audio, 12 streams x 8 words per SNR, rotating noise types)

| SNR | words said | correct events | spurious / wrong events | missed | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| 20 dB | 96 | 96 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| 10 dB | 96 | 95 | 0 | 1 | 1.00 | 0.99 | 0.99 |
| 5 dB | 96 | 76 | 4 | 20 | 0.95 | 0.79 | 0.86 |

- **False alarms on speech-free noise:** 0 events in 900 s (300 s each of white, pink and hum at 10 dB). With zero
  events in 15 minutes, the rule of three puts the 95% upper bound near 12 per hour — a bound, not a rate.
- **Commands:** 40 streams of three commands each (120 commands) per SNR, parsed through the grammar (words inside a command <0.5 s apart, commands >2 s
  apart): **120/120 at 20 dB, 119/120 at 10 dB** commands parsed to the right intent.

### Known limits

- The first 150 ms of a stream must be noise only (the noise floor is initialised from it).
- A sudden, sustained jump in background level is emitted once as a *truncated* segment (always rejected) and
  then absorbed into the floor; it does not retrigger every 1.5 s, but words spoken inside that truncated segment are lost.
- Words are classified one at a time from time-normalised MFCCs; there is no language model and no overlap handling.
- Short words with a silent closure (`play`, `stop`) are the most fragile when a fast speaker shrinks them.
- All speech is synthetic, one fixed vocabulary of 8 words.

## Ideas not done yet

CNN on the time-frequency map (microgrid), a real-speech benchmark (needs a download-free corpus), keyword-spotting
with per-word thresholds from validation data, and a streaming wake-word mode with a rolling posterior smoother.

## Per-word thresholds (calibrate_thresholds)

`calibrate_thresholds(model, X_val, y_val, target_precision=0.95)` picks, per word, the lowest confidence
threshold whose validation precision meets the target (falls back to 0.99 if unreachable). Pass the
resulting dict as `Recognizer(model, threshold=thr)`. Validation data must be disjoint from training data.
No benchmark claim is made yet for how much this improves stream F1 over a single global threshold.
