# Creating a submission

The proxy `train_gpt.py` defaults are set for the proxy (1×A100-40GB, 3 min). Lines that differ for 8×H100 submission are marked with `# SUBMISSION: <value>` comments.

## Proxy → Submission changes

Find all `# SUBMISSION:` comments in `Hyperparameters` and change each default to the submission value:

The code auto-detects:
- Single vs multi-GPU (Parallel Muon handles both)
- FA2 vs FA3 (try/except import, Hopper auto-detected)

If you spot anything else that should be changed for submission, please raise them to the user.

Note: ignore the constraint on number of lines of code for now, it's easy to compress the code file later.

## Features that activate at full budget

Already in the code, only meaningful at ~7000 steps:
- **EMA** (0.997 decay) — averages over full training
- **SWA** (every 50 steps when warmdown < 0.2) — late-stage averaging
- **Late QAT** (STE when warmdown < 0.15) — quantization-aware fine-tuning
- **Warmdown** (3500 iters) — LR ramp, scales with wallclock
- **BigramHash** — zero-init, needs thousands of steps to activate
