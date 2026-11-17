# lora

LoRA in NumPy with manual backprop. It started as a frozen MLP with a low-rank adapter and turned into a small playground for the follow-up ideas: adapters on the attention of the GPT from [`tiny-transformer`](../tiny-transformer), an adapter hub, QLoRA-style quantized bases, LoftQ init and double quantization.

```bash
cd lora
pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q                          # 86 tests, ~30 s
PYTHONPATH=src python -m minilora.cli --sweep               # rank sweep on the toy MLP
PYTHONPATH=src python -m minilora.gpt_demo --rank 4         # adapters on the char GPT, ~20 s
PYTHONPATH=src python -m minilora.hub_demo                  # swap / blend / route adapters
PYTHONPATH=src python -m minilora.qlora_demo                # adapters on an int8/nf4/int4 base
```

## Adapting a char GPT to a new domain

A 62k-parameter char GPT pretrained on Shakespeare, then adapted to synthetic "technical prose" it has never seen. Held-out NLL in nats per char, one seed:

| method | Shakespeare | tech prose | trainable |
|---|---|---|---|
| no adapter | 0.368 | 5.690 | 0 |
| LoRA r=1 | 2.721 | 2.122 | 0.9% |
| LoRA r=4 | 3.234 | 1.504 | 3.7% |
| LoRA r=4, attention + MLP | 4.044 | 0.916 | 9.9% |
| full fine-tune | 5.466 | 0.381 | 100% |

Full fine-tuning fits the new domain best and wrecks the old one. The nice thing with LoRA is that switching the adapter off gives you back the exact base model.

## Other experiments

- Adapter hub: two specialist adapters, each cuts its own domain from ~5.6 to ~1.4 NLL. Routing by scoring a 48-char snippet picked the right one every time. A 50/50 blend lands halfway on both, so it's a compromise, not a union. An SVD-composed adapter reproduces the blend to ~1e-7.
- QLoRA-style: the model gets 4.3x smaller. int8 is essentially lossless. NF4 loses less than plain int4 at the same size (0.373 vs 0.396 on Shakespeare). Adapters trained on the quantized base match the fp32 adapter (1.42-1.45 on tech prose).
- LoftQ init: weight error goes down about 20% vs plain 4-bit at rank 4, but after fine-tuning the inits end up within noise of each other. At 4 bits on a model this small there isn't much to repair.
- Double quantization: storing the block scales as int8 takes NF4 at block 64 from 4.5 to 4.13 bits per weight with identical weight error.

Everything is one tiny model with mostly one seed and synthetic second domains, so treat small differences as noise.
