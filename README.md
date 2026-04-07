# arc

`arc` is the experiment tracker and orchestrator for autoresearch.

## Development

```bash
uv sync
source .venv/bin/activate
arc --help
```

The package is installed in editable mode by `uv sync`, and the `arc`
console script is exposed from the environment's `bin` directory.

If you want `arc` on your shell `PATH` without activating a project
virtualenv, install it as a uv tool:

```bash
uv tool install --editable /Users/scottcui/projects/autoresearch/arc
uv tool update-shell
arc --help
```

## Task-specific commands

When `arc` detects a supported task repository, it exposes additional
task-owned commands.

For Parameter Golf repositories:

```bash
arc run prepare --train-shards 1 --variant sp1024
arc run train
arc run train --gpu H100 --cpu 12 --memory-gb 48
arc run train --gpu H100:8 .arc/submissions/0407/train_gpt_submission.py
arc instruction parameter_golf
```

`prepare` forwards trailing arguments through to the data-preparation
entrypoint, while `train` launches the task's Modal GPU training job.
Both `prepare` and `train` accept `--cpu` and `--memory-gb` overrides,
and `train` also accepts `--gpu`. `arc instruction parameter_golf`
prints the bundled `instructions/base_program.md` file for that task.

## Core workflow

`arc submit <name>` now auto-commits that experiment worktree before
launching it. `arc submit <commit>` still submits an existing tracked
committed node.

Useful follow-up commands:

```bash
arc status
arc tail <commit>
arc result <commit> --verdict=promising|regression|neutral|inconclusive|invalid ...
arc verdict <commit> promising|regression|neutral|inconclusive|invalid
arc fail <commit> ...
arc archive <commit>
```

`arc status` inspects each running node's `run.log` so it can tell you
when a remote run already finished, failed remotely, or lost its log.
Use `arc fail` only for hard execution failures such as crashes, OOMs,
or infra problems. Use `arc result --verdict=invalid` for completed
runs whose reported metric is unusable or disqualified, `neutral` for
flat results, `regression` for worse results, and `inconclusive` when a
run completed but did not cleanly answer the intended question.
`arc archive <commit>` hides stale leaf nodes from the
default tree view without deleting them.
