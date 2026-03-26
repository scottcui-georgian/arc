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
```

`prepare` forwards trailing arguments through to the data-preparation
entrypoint, while `train` launches the task's Modal GPU training job.

## Core workflow

`arc submit <name>` now auto-commits that experiment worktree before
launching it. `arc submit <commit>` still submits an existing tracked
node, and `arc submit --retry <commit>` resubmits a node still marked
`running`.

Useful follow-up commands:

```bash
arc status
arc tail <commit>
arc result <commit> --verdict=promising|unsupported ...
arc fail <commit> ...
arc archive <commit>
```

`arc status` inspects each running node's `run.log` so it can tell you
when a remote run already finished and now needs `arc result` or
`arc fail`. Use `arc fail` only for hard execution failures such as
crashes, OOMs, timeouts, infra problems, or otherwise invalid runs.
`arc archive <commit>` hides stale leaf nodes from the
default tree view without deleting them.
