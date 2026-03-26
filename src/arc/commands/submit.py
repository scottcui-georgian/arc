from __future__ import annotations

import argparse

from arc.app import ArcApp
from arc.commands.base import CommandSpec
from arc.commands.commit import commit_worktree
from arc.errors import ArcError
from arc.executors import load_executor


def register(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        help="Tracked commit hash/prefix, or an experiment name to auto-commit and submit.",
    )
    parser.add_argument(
        "--retry",
        action="store_true",
        help="Allow resubmitting a node currently marked running.",
    )


def run(app: ArcApp, args: argparse.Namespace, extras: list[str]) -> int:
    if extras:
        raise ArcError(f"Unexpected arguments: {' '.join(extras)}")
    app.store.require_initialized()

    auto_committed = False
    record = app.store.get_node_record(args.target)
    if record is None:
        if args.retry:
            raise ArcError("`--retry` only applies to tracked commits already marked running.")
        record = commit_worktree(app, args.target)
        auto_committed = True
    if record.node.archived_at is not None:
        raise ArcError(f"Cannot submit archived node `{record.node.commit}`.")

    allowed_statuses = {"committed"}
    if args.retry:
        allowed_statuses.add("running")
    if record.node.status not in allowed_statuses:
        expected = "`committed`" if not args.retry else "`committed` or `running` with `--retry`"
        raise ArcError(f"Expected status {expected}, got `{record.node.status}`.")

    worktree_root = app.node_worktree_path(record.node)
    log_path = app.node_log_path(record.node)
    result = app.task.submit(record.node, worktree_root, log_path)
    if result is None:
        executor = load_executor()
        result = executor.submit(record.node, log_path)
    app.store.update_node(record.node.commit, status="running")

    if auto_committed:
        print(f"Auto-committed {record.node.name}: {record.node.commit}")
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
