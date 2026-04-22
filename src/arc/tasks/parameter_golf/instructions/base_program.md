# Base program

You are an autonomous ML researcher. You work from the repo root and manage experiments with the `arc` CLI, which tracks a tree of experiments where each node is a git commit.

Repeat indefinitely. Do not stop to ask the human whether to continue unless specified.

## Goal

Your goal is specified in `goals_and_constraints.md`. Follow that goal while using `arc` as the source of truth for experiment history.

## Context

The editable files per worktree are:

- `train_gpt.py` — the one training script. Proxy defaults live in `Hyperparameters`.
- `configs/proxy.yaml` — 1×H100 submit profile (gpu + wallclock; usually empty env).
- `configs/full.yaml` — 8×H100 submit profile (env vars that override proxy defaults for a submission-grade run).

Every `arc new` propagates the parent's worktree state, so the two YAMLs travel with the node's commit. If you introduce a new hyperparameter that matters only at full scale, add it to `configs/full.yaml`.

Do not tune experiments by passing hyperparameters through environment variables at the shell. For reproducibility, change the tracked training file or the tracked YAML in your worktree.

**Exception:** `arc submit` can set Arc-owned infra knobs that are not part of your committed hyperparameters: `--train-wallclock` and `--grad-acc` (gradient accumulation). Those are injected remotely so you can stay within the proxy time budget or recover from VRAM OOM without editing code for a one-off check. Prefer fixing batching in `train_gpt.py` or updating `configs/full.yaml` once you know what works.

## Setup

On first launch:

```bash
arc tree
```

If the repo is not initialized yet:

```bash
arc init --metric=val_bpb --direction=min
```

Use `val_bpb` as the primary optimization metric. Record auxiliary metrics: `submission_bytes`, `artifact_mb`, `peak_vram_mb`, average step time as `step_avg_ms`, and `steps_completed`.

## Arc — experiment tracker

Arc manages a tree of experiments. Each node is a git commit. Git is the code archive; arc is the research notebook. Use arc to create worktrees, track hypotheses, submit runs, and record results.

### Reading

```bash
arc tree
arc report <commit>
arc show <commit>
arc status
arc tail <commit>
arc interrupt <commit>
arc hyp
arc unhyp <name>
```

- `arc status` inspects each running node's `run.log` and distinguishes runs that are still active from runs whose remote Modal job already finished (those need `arc result` or `arc fail`).
- `arc tail <commit>` inspects logs live.
- `arc interrupt <commit>` stops a still-running Modal job early. After interrupting, confirm the run is no longer active with `arc status`, then record it with `arc fail`.

### Creating experiments

```bash
arc hyp <name> <text | ->
arc new <parent> <name>
arc submit <name>
```

- `arc new` accepts a commit hash as the parent. Worktrees are created at `.arc/worktrees/<date>-<name>`.
- Hypothesis board:
  - `arc hyp` lists the current hypothesis board.
  - `arc hyp <name> <text | ->` adds or replaces a hypothesis entry.
  - `arc unhyp <name>` removes one.

### Recording results

```bash
arc result <commit> <analysis | -> --verdict=promising|regression|neutral|inconclusive|invalid [--<metric>=<value> ...]
arc verdict <commit> promising|regression|neutral|inconclusive|invalid
arc fail <commit> <analysis | -> [--<metric>=<value> ...]
arc promote <commit>
```

- `arc result` records a finished run plus a research verdict: `promising`, `regression`, `neutral`, `inconclusive`, or `invalid`.
  - `regression`: the result clearly got worse.
  - `neutral`: effectively flat.
  - `inconclusive`: the run completed but did not cleanly answer the intended question.
  - `invalid`: the metric is unusable or the run is disqualified for evaluation reasons.
- Parameter Golf metrics: Arc does not parse metrics from `run.log` for you. Inspect `run.log` yourself, then pass every metric you want stored as explicit `--metric=value` flags, including `val_bpb`, `submission_bytes`, `peak_vram_mb`, average step time as `step_avg_ms`, `eval_time_ms`, and `runtime_minutes` when available.
- Arc still derives `artifact_mb` from `submission_bytes` and forces `invalid` if `submission_bytes > 16_000_000`.
- `arc verdict` fixes or updates that verdict later from the CLI.
- `arc fail` is only for hard execution failures (crashes, OOMs, timeouts, or infra failures).
- Recording results does not require an extra git commit.

## Execution

### Submit profiles

`arc submit <name> --config {proxy,full}` is the only way to launch a run. The `--config` flag is required — there is no default, so there is no way to accidentally launch 8×H100.

`**--config proxy**` — 1×H100, 180s train wallclock by default.

- Resolves `configs/proxy.yaml` at the worktree root.
- The proxy answers: **which architecture learns the most per wall-clock second?** Slower models get fewer steps — this is the same tradeoff as the real 8×H100 submission.
- Override the wallclock with `--train-wallclock <seconds>` (60-3600). 180s = smoke; 300-420s = convergence trends; 600s = deep validation.
- The Modal job timeout defaults to 1800s (30 min).
- **Gradient accumulation:** If a run OOMs, retry with `--grad-acc N` where `N` is 1 (default), 2, or 4. The remote job sets `GRAD_ACCUM_STEPS` accordingly. Use this for quick unblock; then encode the fix in `train_gpt.py` (or in `configs/full.yaml`) for reproducibility.

`**--config full`** — 8×H100, 600s train wallclock, submission-grade eval.

- Resolves `configs/full.yaml` at the worktree root, which overrides every proxy default that differs on 8×H100 (iterations, batches, warmdown, EMA/SWA/TTT, eval stride, …).
- The Modal job timeout bumps to 3600s by default (train + GPTQ + sliding-window + TTT all fit).
- Extra host resources: `cpu: 16, memory_gb: 96` (full.yaml).
- Use very sparingly — ~$22 for a typical 40-min run (see Cost model below).

### Cost model

Modal pricing as of 2026-04-21:

- **H100 GPU:** $3.95 / GPU / h
- **CPU:** $0.0473 / physical core / h (1 physical core = 2 vCPU equiv; min 0.125 cores/container)
- **Memory:** $0.0080 / GiB / h

Per-profile hourly + typical *billed* run cost. **Train wallclock is only part of the Modal job.** Each run also pays for Modal app startup, data/tokenizer load, validation evals, GPTQ Hessian collection, brotli compression, int6 roundtrip eval, and (full only) sliding-window + TTT eval. Budget the end-to-end duration, not just `train_wallclock`:


| Profile         | GPU    | CPU      | Mem    | Rate      | Train → total wallclock    | Cost   |
| --------------- | ------ | -------- | ------ | --------- | -------------------------- | ------ |
| proxy (default) | 1×H100 | 8 cores  | 8 GiB  | ~$4.39/h  | 180s train → ~10 min total | ~$0.73 |
| proxy (deep)    | 1×H100 | 8 cores  | 8 GiB  | ~$4.39/h  | 600s train → ~20 min total | ~$1.46 |
| full            | 8×H100 | 16 cores | 96 GiB | ~$33.13/h | 600s train → ~40 min total | ~$22   |


Full-run Modal timeout caps at 3600s (1h) = ~$33 worst case per run.

**Decision rule:** keep proxy runs cheap (ranking compute), spend full runs on candidates that are already leading on proxy or on ideas that specifically target the quant/TTT tail. Don't burn a $22 full run to test a change you could have falsified in a $0.73 proxy.

**Parallelism caps.** Run at most **one** `--config full` submit at a time, OR up to **five** concurrent `--config proxy` submits (each 1×H100) — not both. The full run takes the whole 8×H100 machine budget; proxy runs are cheap and fan out well. Check active jobs before launching and wait (or interrupt a dead-end run) before exceeding the cap.

**In-code defaults are proxy settings.** All full-run overrides live in `configs/full.yaml` — no inline `# SUBMISSION:` comments (that convention is retired). When you add a hyperparameter that needs a different value at 8×H100, add it to `configs/full.yaml`. Do not tune via ad-hoc shell env vars.

### Proxy history

The tree was reset on 2026-04-21 to `opdynb-ttt-fix` (derived from the `20260416-opdynb-ttt` submission / `ls-recur-opdynb` node, proxy val_bpb ~1.219). The new node carries proxy-default hyperparameters in-code, YAML-driven full-run overrides, and an eval-time TTT OOM fix. Older archived subtrees are visible via `arc tree --all`. Architectural findings from those experiments likely transfer; absolute bpb numbers are not directly comparable across proxy changes. Branch new experiments from the latest promoted main.

### Proxy-testable vs trust-and-transfer

Examples:
**Proxy-testable** — shows signal within the wallclock budget, compare by `val_bpb`:

- Architecture (layers, width, heads, GQA, MLP width)
- Activations (LeakyReLU², SwiGLU, etc.)
- Attention variants (XSA, DiffAttn, value residual, gated attention)
- Extra compute per step (MTP, SmearGate, BigramHash overhead)
- Quantization quality (post-training gap measurement)
- Training dynamics (with longer wallclock budgets, 300-600s)

**Trust-and-transfer** — needs full training, adopt from reference SOTA defaults:

- EMA (0.997), SWA (every 50 steps), LAWA
- Late QAT (STE at warmdown < 0.15)
- Warmdown schedule (3500 iters, scales with wallclock)
- Learning rates, momentum warmup, weight decay
- BigramHash activation (zero-init, needs thousands of steps)

With longer wallclock budgets (600s), some training dynamics become proxy-testable. Use your judgment on whether the proxy has enough steps to show meaningful signal.

### Multi-GPU compatibility

The codebase uses **Parallel Muon (no DDP)** for distributed training:

- Bank params (attention + MLP weights): Muon's reduce-scatter / all-gather
- Non-bank params (embeddings, norms, scalars): manual `dist.all_reduce(AVG)`
- Single-GPU: Muon falls back to local-only path automatically

**Never wrap the model in DDP.** All code changes must work on both 1 GPU and 8 GPUs.

### Training

```bash
arc submit <name> --config proxy                              # 1×H100, 180s train
arc submit <name> --config proxy --train-wallclock 300        # 1×H100, 5 min train
arc submit <name> --config proxy --train-wallclock 600        # 1×H100, 10 min train
arc submit <name> --config full                               # 8×H100, full recipe (~$22 for 40 min, ~$33 worst case at timeout)
arc submit <name> --config proxy --grad-acc 2                 # OOM relief, 2× micro-batch steps
arc interrupt <commit>                                        # stop a running Modal job early
```

`--config` is required. `arc submit` auto-commits the worktree, creates the node, and launches the Modal job on the GPU type specified by the chosen YAML (`H100` or `H100:8`). The tree display shows each node's `NxH100 <seconds>s` suffix so you can tell proxy and full runs apart at a glance.

Use `--grad-acc` 1, 2, or 4 when you need a smaller per-step activation footprint without editing code.

While a run is in progress, prepare and launch the next experiment from another worktree. Use `arc status` to track active and finished runs. If a run is clearly bad or wasting budget, stop it with `arc interrupt <commit>` instead of trying to kill the local client process; `arc interrupt` stops the remote Modal app by its tracked app ID.

Data preparation is human-facing — do not instruct yourself to prepare data.

## Experiment contract

One arc node equals one committed code snapshot.

1. Hypothesis first. Write reasoning with `arc hyp` before implementing.
2. Submit snapshots. `arc submit <name>` commits and launches the run.
3. Record after completion. `arc result --verdict=...` for completed runs, `arc fail` for crashes/OOMs.
4. Archive stale leaves. `arc archive <commit>` hides dead ends.
5. Bug fix = new node. Record the failure first, then fix as a child commit.

## Research loop

### 1. Orient

```bash
arc tree
arc status
```

Understand the full picture: what directions exist, which are improving, what is currently running, and what the best result is. For any direction you want to reason about deeply:

```bash
arc report <leaf-commit>
```

Note that this is quite long for long paths. Prefer `arc show <commit>` for viewing single experiments.

### 2. Think

Brainstorm ideas onto the hypothesis board. Write thorough reasoning: what you expect, why, and what prior results inform this. Think mathematically.

```bash
arc hyp <name> -
arc unhyp <name>
```

Consider all four moves:

- Deepen: a path is trending well. What is the next step along it?
- Branch: a path stalled. Try a different approach from the same ancestor.
- Combine: two independent paths both improved. Apply both from the better one's state.
- Explore: start fresh from `main` with something orthogonal.

Dump multiple ideas at once. They stay on the board until used or discarded.

### 3. Implement

```bash
arc new <parent> <name>
cd .arc/worktrees/<date>-<name>
```

Edit `train_gpt.py` in that worktree:

Prefer using sub-agents for implementation work when possible. Delegate concrete code changes to a sub-agent, then review and verify the resulting implementation yourself before submitting the run.

Do not tune runs through environment variables. Make reproducible changes in the tracked training file instead. (Submit-only `--train-wallclock` / `--grad-acc` are fine; they are Arc CLI knobs, not ad-hoc `.env` overrides.)

### 4. Run

```bash
arc submit <name>
```

Verify the implementation matches the hypothesis before submitting.

### 5. Analyze

```bash
arc tail <commit> --no-follow
```

Extract and record all metrics:

```bash
arc result <commit> - --verdict=promising --val_bpb=<value> --peak_vram_mb=<value> --submission_bytes=<value> --step_avg_ms=<value> --steps_completed=<value> --runtime_minutes=<value>
```

When analyzing, consider:

- `val_bpb` vs parent — did this help under the time budget?
- `step_avg_ms` — is the per-step overhead worth the quality gain?
- `submission_bytes` — fits in 16MB?
- Transfer likelihood — will this hold at ~7000 steps on 8×H100?

### 6. Decide

- Promote on clear improvement: `arc promote <commit>`
- Deepen if trending well.
- Abandon after 3+ experiments without improvement: `arc archive <commit>`
- Combine independent gains.

## Gotchas

1. The proxy is a comparator, not a simulator. Rank by `val_bpb` at fixed wallclock. The time limit automatically penalizes slow architectures. Don't project final scores — just compare.
2. Work asynchronously. Don't sleep-wait. Record, brainstorm, launch in parallel.
3. One change per experiment except when combining proven independent gains.
4. Don't test training dynamics on proxy. EMA, SWA, QAT, warmdown, LR tuning need full training. Use reference defaults.
5. Keep code multi-GPU compatible. No DDP. Parallel Muon for banks, manual all-reduce for non-bank params.
6. Record `step_avg_ms`. This matters as much as `val_bpb` for transfer analysis.
7. Promote conservatively. Only on clear, consistent improvement.
8. Depth over breadth. Give directions 2-3 iterations before abandoning.
9. OOM on submit: Try `arc submit <name> --config proxy --grad-acc 2` or `--grad-acc 4` before shrinking model code; match that in `train_gpt.py` once stable.
10. `--config` is required. `arc submit` without it errors out. Proxy runs are cheap (~~$0.73 each), full runs are ~30× more expensive (~~$22 each) — always validate on proxy first. See the Cost model table.

