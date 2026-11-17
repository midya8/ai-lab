# Roadmap

Things I want to try next, roughly by project. The history of what's already done is in the git log and each project's README.

## Models and training

- **tiny-transformer:** a proper key-value recall benchmark with a bigger model, to test long-context use for real instead of on a memorised corpus.
- **lora:** 2-bit bases to find where LoftQ init actually starts to matter, and streaming adapters in and out at serve time.
- **microgrid:** MoE and Hopfield layers on top of the autodiff engine.
- **flash-attention:** per-channel key quantization with a full-precision residual block, prefix caching with a radix tree, a batch dimension.
- **mixture-of-experts:** a task with structure shared across clusters, to give shared experts a fair test. MoE inside the tiny GPT's FFN.
- **mamba-numpy:** the associative recall study, HiPPO-style init, and a head-to-head with tiny-transformer on the same tasks.
- **specdec:** a KV cache, so batched verification can show a real wall-clock number.
- **diffusion-2d:** actually train with min-SNR weighting instead of just having the conversions.
- **normalizing-flows:** spline couplings, and a mixture base distribution to get around the topology cost on the ring.
- **gnn:** sweep APPNP's teleport per homophily level, sparse operators, heterophily-aware layers.
- **cnn-explain:** tighter bounds (CROWN) to certify the FGSM-trained net, brightness/offset augmentation.
- **mcts:** Connect-Four support for the AlphaZero-lite net, and a conv policy on microgrid.
- **rlgrid:** probably done. The next RL thing should be a new project.

## Understanding models

- **mech-interp:** figure out what the layer-0 MLP computes, and train an attention-only model to see if a classic induction head shows up.
- **sparse-autoencoders:** a better absorption metric for correlated features, and the regime with more features than latents.
- **hopfield:** a Hopfield pooling layer, and checking the measured beta transition against the separation bound.
- **ntk:** finite-width corrections, double descent in n, a mu-P comparison.
- **gaussian-processes:** learned inducing locations, and using it as the surrogate inside bayes-opt.
- **bayes-opt:** gradient-based acquisition optimisation, and a proper multi-seed comparison for MES.
- **conformal:** a covariate shift that moves points into the noisy region, the case that actually hurts.
- **datadoctor:** a real noisy-label dataset that doesn't need a download.

## LLM tooling

- **rag-evals / docqa / codesearch:** try real embeddings against the lexical baselines on the same query sets.
- **reviewkit:** a fresh held-out set (the current one is now tuned on), and running the Action on a real PR.
- **codesearch:** a third query set written by someone other than me.
- **ticket-triage:** feed live router output into the SLA simulation.
- **voicekit:** a real-speech benchmark, and a better VAD below 5 dB.
