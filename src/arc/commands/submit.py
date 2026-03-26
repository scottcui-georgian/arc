from __future__ import annotations

import argparse
from pathlib import Path

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.errors import ArcError
from arc.executors import load_executor


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("commit", help="Experiment commit hash or prefix.")
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Allow resubmitting a node currently marked running.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    record = app.store.get_node_record(args.commit)
    if record is None:
        raise ArcError(f"Unknown commit: {args.commit}")
    allowed_statuses = {"committed"}
    if args.retry:
        allowed_statuses.add("running")
    if record.node.status not in allowed_statuses:
        expected = "`committed`" if not args.retry else "`committed` or `running` with `--retry`"
        raise ArcError(f"Expected status {expected}, got `{record.node.status}`.")

    worktree_root = Path(app.paths.repo_root, record.node.worktree)
    log_path = worktree_root / "run.log"
    result = app.task.submit(record.node, worktree_root, log_path)
    if result is None:
        executor = load_executor()
        result = executor.submit(record.node, log_path)
    app.store.update_node(record.node.commit, status="running")

    print(f"Submitted {record.node.commit} ({record.node.name})")
    print(f"Status: {record.node.status} → running")
    print(f"Executor: {result.backend}")
    print(f"Log:      {app.relative_path(result.log_path)}")
    if result.process_id is not None:
        print(f"PID:      {result.process_id}")
    return 0


COMMAND = CommandSpec(
    name="submit",
    help="Submit an experiment for execution.",
    register=register,
    run=run,
)
