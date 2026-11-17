# voicekit

An offline voice-command pipeline in NumPy: a streaming voice-activity detector, MFCC features, two word classifiers (a hand-backpropped MLP and a DTW template matcher), open-set rejection for words it doesn't know, a wake word gate, and a small command grammar. No audio libraries and no model downloads.

One important caveat up front: there's no real speech here. Words are synthesised from formant-style phonemes with random pitch, vocal tract length and speaking rate, plus white, pink or hum noise at a set SNR. It's a testbed for the engineering (VAD, streaming, rejection, evaluation), and the numbers say nothing about human voices.

```bash
cd voicekit
pip install -r requirements.txt
PYTHONPATH=src python -m voicekit demo            # trains in ~15 s, then recognises a noisy command stream
PYTHONPATH=src python -m voicekit synth lights --out lights.wav --snr 10
PYTHONPATH=src python -m voicekit segment lights.wav
PYTHONPATH=src python experiments/evaluate.py     # ~1 min
PYTHONPATH=src python -m pytest -q
```

## Results

Unseen synthetic speakers and noise, 270 test utterances per row:

| SNR | VAD misses | MLP end-to-end | DTW (detected only) | unknown words rejected |
|---|---|---|---|---|
| 30 dB | 0% | 0.978 | 0.933 | 0.87 |
| 10 dB | 0.4% | 0.956 | 0.926 | 0.70 |
| 5 dB | 11% | 0.841 | 0.833 | 0.93 |
| 0 dB | 43% | 0.519 | 0.856 | 0.88 |

- From 10 dB up it's dependable. Below 5 dB the VAD is the bottleneck, not the classifier.
- Spectral subtraction didn't help, since the MLP was already trained on noisy audio.
- The MLP beats DTW at every SNR.
- Open-set rejection is the weakest part: 70-93% of unknown words get rejected and the rest are confidently mislabelled.
- Speakers with vocal tracts outside the training range drop accuracy from 0.974 to 0.881.

Streaming at 10 dB: 95 of 96 words right and no false alarms. At 5 dB recall falls to 0.79. There were zero events in 15 minutes of pure noise. 119 of 120 full commands parsed correctly at 10 dB. Per-word confidence thresholds didn't beat one global threshold.

Not every demo seed is clean. With `--seed 3` a fast speaker's "play" gets clipped to 140 ms and is confidently read as "lights". That's the failure mode thresholds can't catch.
