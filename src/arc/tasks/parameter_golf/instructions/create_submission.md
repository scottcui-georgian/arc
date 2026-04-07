# Creating a submission

The proxy `train_gpt.py` defaults are set for the proxy (1×A100-40GB, 3 min). Lines that differ for 8×H100 submission are marked with `# SUBMISSION: <value>` comments.

## Proxy → Submission changes

Find all `# SUBMISSION:` comments in `Hyperparameters` and change each default to the submission value:

| Setting | Proxy default | Submission value |
|---------|---------------|------------------|
| `train_batch_tokens` | 262,144 | 786,432 |
| `max_wallclock_seconds` | 180.0 | 600.0 |
| `warmdown_iters` | 0 | 3500 |
| `muon_momentum_warmup_steps` | 20 | 1500 |
| `val_loss_every` | 0 | 4000 |
| `train_log_every` | 20 | 500 |
| `eval_stride` | 0 | 64 |
| `ema_enabled` | 0 | 1 |
| `swa_enabled` | 0 | 1 |
| `late_qat_threshold` | 0.0 | 0.15 |
| `gptq_calib_seqs` | 16 | 64 |

No other code changes needed. The code auto-detects:
- Single vs multi-GPU (Parallel Muon handles both)
- FA2 vs FA3 (try/except import, Hopper auto-detected)

## Features that activate at full budget

Already in the code, only meaningful at ~7000 steps:
- **EMA** (0.997 decay) — averages over full training
- **SWA** (every 50 steps when warmdown < 0.2) — late-stage averaging
- **Late QAT** (STE when warmdown < 0.15) — quantization-aware fine-tuning
- **Warmdown** (3500 iters) — LR ramp, scales with wallclock
- **BigramHash** — zero-init, needs thousands of steps to activate

## Running the submission

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation (3 seeds for statistical confidence)

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=314 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=999 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Expected inter-seed std: ~0.0005–0.0015 BPB. A new record requires ≥0.005 nats improvement with p < 0.01.

## What to watch for on 8×H100

- **Step time**: reference SOTA gets ~87 ms/step. Much slower = architecture overhead problem.
- **Loss curve**: should drop rapidly (6.9 → 2.4 in 500 steps). Stalling = gradient sync bug.
- **Artifact size**: must be ≤16 MB after GPTQ + compression.
- **Eval time**: standard eval + sliding window must finish within 10 min eval budget.
