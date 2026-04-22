# Creating a submission

A "submission" is the committed code + configs for one node, run on 8×H100 via
`arc submit <name> --config full`. The same worktree that passes the proxy smoke
test (`--config proxy`) produces the submission artifact — no code swap, no
`.arc/submissions/<folder>/` duplication.

## Workflow

1. Work in the worktree created by `arc new <parent> <name>`.
2. Edit `train_gpt.py` using proxy defaults (1×H100).
3. Verify with `arc submit <name> --config proxy` (180s, 1×H100).
4. If the proxy result is promising, launch the submission-grade run with
   `arc submit <name> --config full` (600s train + full eval, 8×H100).
5. Record both via `arc result`. Promote on clear improvement.

## YAML configs

Each worktree ships two files at its root:

- `configs/proxy.yaml` — single H100, 180s wallclock. Empty `env:` — the
  `train_gpt.py` defaults are the proxy values.
- `configs/full.yaml` — 8×H100, 600s wallclock, and every env-var override
  needed to turn the proxy recipe into the submission-grade recipe (larger
  batches, longer warmdown, EMA/SWA/TTT on, evaluation stride, etc.).

The two files travel with the worktree's git commit, so every experiment has
its own configs. When you introduce a new hyperparameter that should differ at
full scale, add it to `configs/full.yaml` — do NOT add `# SUBMISSION:` inline
comments (that convention is retired).

## Features that only activate at full budget

Proxy runs don't exercise these; the full YAML switches them on:

- **EMA** (0.997 decay) — averages over full training
- **SWA** (every 50 steps when warmdown < 0.2) — late-stage averaging
- **Late QAT** (STE when warmdown < 0.15) — quantization-aware fine-tuning
- **Warmdown** (3300 iters) — LR ramp, scales with wallclock
- **BigramHash** — zero-init, needs thousands of steps to activate
- **TTT** — test-time training during eval; needs `EVAL_STRIDE > 0`
- **Sliding-window eval** — keyed off `EVAL_STRIDE > 0`

## Multi-GPU correctness

`arc submit --config full` requests `H100:8` from Modal, which runs training
under `torch.distributed.run` with Parallel Muon. The in-code path must work
both at `world_size=1` (proxy) and `world_size=8` (full). Never wrap the model
in DDP; the Muon optimizer + manual all-reduces in `train_gpt.py` cover the
distributed path.

## Artifact size limit

The packaged submission must stay under 16 MB. Arc forces `verdict=invalid` when
`submission_bytes > 16,000,000`. Compression knob is `BROTLI_QUALITY` in
`full.yaml` (default 9 for faster compress; raise to 11 for a small size win if
you're near the limit).
