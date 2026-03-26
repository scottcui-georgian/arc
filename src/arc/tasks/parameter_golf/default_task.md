# Default task — autonomous experiment loop

## Setup

On first launch:
```bash
arc tree
```

If not init yet,
```bash
arc init --metric=val_bpb --direction=min
```

## Loop

Repeat indefinitely. Do not stop to ask the human whether to continue.

### 1. Orient

```bash
arc tree
arc status
```

Understand the full picture: what directions exist, which are improving, what's currently running, what's the best result. For any direction you want to reason about deeply:

```bash
arc report <leaf-commit>
```

### 2. Think

Brainstorm ideas onto the hypothesis board. Write thorough reasoning — what you expect, why, and what prior results inform this. Think mathematically.

```bash
arc hyp <n> - <<'EOF'
...few paragraphs of reasoning...
EOF
```

Consider all four moves:

- **Deepen**: a path is trending well → what's the next step along it?
- **Branch**: a path stalled → try a different approach from the same ancestor.
- **Combine**: two independent paths both improved → apply both from the better one's state.
- **Explore**: start fresh from main with something orthogonal.

Dump multiple ideas at once. They stay on the board until used or discarded.

### 3. Implement

Pick an idea from the board:

```bash
arc new <parent> <n>
cd .arc/worktrees/<date>-<n>
```

Edit the task training file in that worktree:

- `train_gpt.py` in the newer root-level layout

Do not tune runs through environment variables. Make reproducible changes in the tracked training file instead.

### 4. Run

```bash
arc commit <n>
arc submit <commit>
```

`arc submit` is the tracked execution command. It launches the Modal-backed train job for that worktree and appends output to `<worktree>/run.log`.

If data preparation is needed, stop and seek for help.

Treat the single-A10 run as a proxy for the real 8xH100 target. Prefer changes that are likely to transfer to the final submission setting, and avoid tuning specifically for quirks of this proxy.
Do not treat A10 training or evaluation wallclock as hard local pass/fail gates. They are directional signals for the final 8xH100 run. Artifact bytes still matter directly.

While a run is in progress, you can prepare and launch the next experiment from another worktree. Use `arc status` to see which nodes arc still considers `running`, but do not assume that means arc auto-detected remote completion. To learn whether a run actually finished, inspect its `run.log`.

### 5. Analyze

When a run finishes:

```bash
cat .arc/worktrees/<date>-<n>/run.log
```

Record with thorough analysis — what happened, why, and what it means for next steps:

```bash
arc result <commit> - --val_bpb=<value> --peak_vram_mb=<value> <<'EOF'
...few paragraphs of analysis...
EOF
```

In that analysis, reason about the real submission objective, not just the proxy score. At minimum, keep track of final roundtrip `val_bpb`, likely 8xH100 training behavior, likely 8xH100 evaluation behavior, and artifact bytes.

For crashes or timeouts:

```bash
arc fail <commit> - --peak_vram_mb=<value> <<'EOF'
...what went wrong, whether the idea is worth retrying...
EOF
```

If a crash was an obvious bug, fix it in the same worktree, commit as a new node (child of the failed one), and rerun. A small number of retries per idea is fine.

### 6. Decide

After recording results:

- **Promote** if a node is the new best and the improvement is clear: `arc promote <commit>`
- **Deepen** if the direction is trending well — go brainstorm the next step.
- **Abandon** if 3+ experiments on a path haven't improved. The nodes stay in the tree for reference.
- **Combine** if two directions both show independent gains.

Then go back to step 1.

## Principles

- **Depth over breadth.** A bad first result often just needs a follow-up (LR adjustment, init change). Give directions 2–3 iterations before giving up.
- **The tree is your memory.** Use `arc report` to reload context for any direction. Use `arc tree` for the big picture. Don't try to hold everything in your head.
- **One change per experiment.** If you change two things, you can't tell which mattered.
- **Record everything.** Failures are data. They tell future iterations what doesn't work.
- **Promote conservatively.** Only on clear, consistent improvement — not a single marginal gain.
