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
