Please create the submission version of `tran_gpt.py` named `train_gpt_submission.py` in the same dir as `train_gpt.py`. Don't need to commit.

`train_gpt_submission.py` will be running on 8*H100 with a concrete 10m training budget, so please use flash attention 3 and adjust hypermaraters and implement techniques that are sensitive to iterations and are great but discarded because of the proxy.

For tunning, you can refer to some existing leaderboard submissions at "/Users/scottcui/projects/autoresearch/parameter_golf-reference/records/track_10min_16mb".

Report tables of hyperparameter tuned and techniques implemented for the submission after the implementation.

Some techniques for full run might be:
- EMA + SWA — Reference submissions use EMA(0.997) + SWA(every 50 steps). At 500 proxy steps, EMA drags toward init (+0.25 bpb regression). At 20K steps, it stabilizes and improves generalization.
- Late QAT — Quantization-aware training in the final phase. Add quantization noise to weights during forward pass when loss is below a threshold (0.15 in reference). Trains the model to be robust to int5 quantization. Free improvement at full scale, meaningless at proxy.
- BigramHash — Learnable bigram feature table. Zero-initialized, needs many steps to activate. Reference uses BigramHash(3072) or (1536). Showed proxy regression but present in all top submissions. ~0.01-0.02 bpb at full scale
- Currently we use 32 training batches for Hessian collection. Reference ValCalib uses autoregressive self-generated calibration — generates 64 seqs × 2048 tokens from the model itself (temp=0.8). This is more principled (matches inference distribution) and avoids accessing training data during eval. The reference already has a generate_autoregressive_calib function we can port.

Training schedule:
- iterations: 500 → 10000
- max_wallclock_seconds: 0 → 600
- train_batch_tokens: 262,144 → 524,288
- warmdown_iters: 30 → 3500
- muon_momentum_warmup_steps: 20 → 500

Learning rates:
- matrix_lr: 0.04 → 0.025
- scalar_lr: 0.04 → 0.025
- embed_lr: 0.6 → 0.6 (unchanged)
- head_lr: 0.008 → 0.008 (unchanged)

Optimizer:
- muon_momentum_warmup_start: 0.85 → 0.92
- muon_momentum_warmup_steps: 20 → 1500
- grad_clip_norm: 0.0 → 0.3
- muon_wd: (add) 0.04
- adam_wd: (add) 0.04

Architecture tweaks:
- xsa_last_n: 4 → 11 (all layers)

New techniques to add:
- EMA: decay=0.997
- SWA: every 50 steps
- BigramHash: vocab=3072, dim=112
- Late QAT: threshold=0.15
- GPTQ calib batches: 32 → 256

Eval:
- Sliding window eval: stride=64 (post-quant only)

New techniques:
- EMA: 0.997
- SWA: every 50
- BigramHash: vocab=3072, dim=112
- Late QAT: threshold=0.15
- GPTQ calib batches: 32 → 256