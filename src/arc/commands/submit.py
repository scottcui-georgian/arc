from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.commands.commit import commit_worktree
from arc.errors import ArcError
from arc.executors import load_executor
from arc.tasks.base import SubmitOutcome


def _train_wallclock_type(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}")
    if seconds < 60 or seconds > 3600:
        raise argparse.ArgumentTypeError("train wallclock must be between 60 and 3600 seconds")
    return seconds


def _grad_accum_type(value: str) -> int:
    try:
        steps = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer value: {value!r}")
    if steps not in (1, 2, 4):
        raise argparse.ArgumentTypeError("grad accum must be 1, 2, or 4")
    return steps


def register(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Submit a tracked experiment for execution. Passing an experiment name instead of a "
        "tracked commit auto-commits that worktree first."
    )
    parser.add_argument(
        "target",
        help="Tracked commit hash/prefix, or an experiment name to auto-commit and submit.",
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=("proxy", "full"),
        help="Submit profile. Resolves `configs/{proxy,full}.yaml` at the worktree root. "
        "`proxy` = 1×H100 smoke. `full` = 8×H100 submission-grade run.",
    )
    parser.add_argument(
        "--train-wallclock",
        type=_train_wallclock_type,
        default=None,
        metavar="SECONDS",
        help="Override the profile's train wallclock budget (60-3600s). "
        "Prefer editing the YAML for reproducibility.",
    )
    parser.add_argument(
        "--grad-acc",
        type=_grad_accum_type,
        default=None,
        metavar="N",
        help="Gradient accumulation steps (1, 2, or 4). Default 1. "
        "Increases micro-batch divisor — use 2 or 4 to unblock OOM without editing code.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    auto_committed = False
    record = app.store.get_node_record(args.target)
    if record is None:
        record = commit_worktree(app, args.target)
        auto_committed = True
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot submit archived node `{record.node.commit}`.")

    if record.node.status not in ("committed", "completed"):
        raise ArcError(
            f"Expected status `committed` or `completed`, got `{record.node.status}`."
        )

    worktree_root = app.node_worktree_path(record.node)
    log_path = app.node_log_path(record.node)
    train_wallclock = getattr(args, "train_wallclock", None)
    grad_acc = getattr(args, "grad_acc", None)
    config_name = getattr(args, "config", None)
    outcome = app.task.submit(
        record.node,
        worktree_root,
        log_path,
        config_name=config_name,
        train_wallclock=train_wallclock,
        grad_accum_steps=grad_acc,
    )
    result = None
    submit_metrics: dict[str, float] = {}
    if isinstance(outcome, SubmitOutcome):
        result = outcome.result
        submit_metrics = dict(outcome.metrics)
    elif outcome is not None:
        result = outcome
    if result is None:
        executor = load_executor()
        result = executor.submit(record.node, log_path)
    if submit_metrics:
        app.store.upsert_metrics(record.node.commit, submit_metrics)
    app.store.update_node(record.node.commit, status="running")

    if auto_committed:
        print(f"Auto-committed {record.node.name}: {app.display_commit(record.node.commit)}")
    print(f"Submitted {app.display_commit(record.node.commit)} ({record.node.name})")
    print(f"Status: {record.node.status} → running")
    print(f"Executor: {result.backend}")
    print(f"Log:      {app.relative_path(result.log_path)}")
    if result.process_id is not None:
        print(f"PID:      {result.process_id}")
    return 0


COMMAND = CommandSpec(
    name="submit",
    help="Submit an experiment, auto-committing by name when needed.",
    register=register,
    run=run,
)
