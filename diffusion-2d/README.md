# diffusion-2d

A small diffusion model (DDPM, DDIM, classifier-free guidance, a DPM-Solver-2 sampler) in NumPy with hand-written backprop, trained on 2-D data.

The data is a mixture of 8 Gaussians on a ring, which means the optimal noise predictor is known in closed form. So instead of eyeballing samples I can measure how far the trained net is from the best possible model, and run each sampler with a perfect model to separate sampler error from learning error.

```bash
cd diffusion-2d
pip install -r requirements.txt
PYTHONPATH=src python -m diffusion2d --steps 4000   # trains in ~15 s, prints tables and an ASCII scatter
PYTHONPATH=src python -m pytest -q
python experiments/run_all.py                       # the numbers below, 3 seeds, ~4 min
```

The net is a 36.7k-parameter MLP with sinusoidal time features and a class embedding.

## Results

Distance to the true distribution (sliced-W1; two exact sample sets are 0.027 apart, so that's the floor):

| steps | sampler | trained net | exact model |
|--:|--|--:|--:|
| 100 | DDPM | 0.045 | 0.029 |
| 50 | DDIM | 0.045 | 0.037 |
| 20 | DDIM | 0.066 | 0.056 |
| 10 | DDIM | 0.103 | 0.091 |
| 5 | DDIM | 0.187 | 0.216 |

With a perfect model DDPM sits right at the floor. The trained net adds a bit on top. Below ~20 steps the sampler itself is the main source of error, and deterministic DDIM beats stochastic sampling at every reduced step count.

Where the net is wrong: its excess error is biggest at low noise (t around 12), where it has to resolve the narrow modes. The training loss (~0.13) shows none of this because it's dominated by the irreducible part.

Guidance: conditional accuracy without guidance is 0.963, which matches the exact model's ceiling (neighbouring modes overlap). Guidance scale 2 gets 100% by shrinking the spread. At scales 4-8 the exact sampler keeps tightening but the trained net starts spreading out again, since it's being pushed far outside what it saw in training.

Smaller follow-ups: DPM-Solver-2 loses to DDIM below ~12 function evaluations on this coarse 100-step schedule and wins from 16 up. Plain v-prediction was very slightly better than eps-prediction.

## Limitations

This is 2-D toy data. It shows the machinery and where the error budget goes, not anything about image quality. Data is sampled fresh every step, so there's no memorisation study. Three seeds, single seed for the follow-ups.
